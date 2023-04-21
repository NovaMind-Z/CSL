import numpy as np
import torch
import json
import os
from tqdm import tqdm
import random
import framework.ops
import seaborn as sns
import matplotlib.pyplot as plt
from torch import optim

import t2vretrieval.encoders.text_encoder
import t2vretrieval.encoders.video_encoder

import t2vretrieval.models.globalmatch
from t2vretrieval.models.criterion import cosine_sim

from t2vretrieval.models.globalmatch import VISENC, TXTENC
from t2vretrieval.models.SymbolGenerateModel import SymbolGenerateModel
from t2vretrieval.models.MemorySelectModel import MemorySelectModel
from t2vretrieval.models.LanguageModelCriterion import LanguageModelCriterion
from t2vretrieval import evaluation


VISDEC = 'memory_based_selector'
TXTDEC = 'symbolic_generator'
def diff(a, l):
    n = a.size(0)
    m = a.size(1)
    b = a.reshape(n*m, 1)
    c = b.repeat(1, l)
    d = c.reshape(n, m*l)
    return d

class CSLModelConfig(t2vretrieval.models.globalmatch.GlobalMatchModelConfig):
    def __init__(self):
        super().__init__()

        self.hard_topk = 1
        self.max_violation = True

        self.loss_weights = None

        self.subcfgs[VISENC] = t2vretrieval.encoders.video_encoder.VideoEncoderConfig()
        self.subcfgs[TXTENC] = t2vretrieval.encoders.text_encoder.TextEncoderConfig()
        self.subcfgs[VISDEC] = t2vretrieval.models.SymbolGenerateModel.SymbolGenerateModelConfig()
        self.subcfgs[TXTDEC] = t2vretrieval.models.MemorySelectModel.MemorySelectModelConfig()





class CLSModel(t2vretrieval.models.globalmatch.GlobalMatchModel):
    def __init__(self, config, _logger=None, gpu_id=0):
        super().__init__(config, _logger, gpu_id)
        self.crit = LanguageModelCriterion()

        self.params, self.optimizer, self.lr_scheduler = self.build_optimizer()

        num_params, num_weights = 0, 0
        for key, submod in self.submods.items():
            for varname, varvalue in submod.state_dict().items():
                self.print_fn('%s: %s, shape=%s, num:%d' % (
                    key, varname, str(varvalue.size()), np.prod(varvalue.size())))
                num_params += 1
                num_weights += np.prod(varvalue.size())
        self.print_fn('num params %d, num weights %d' % (num_params, num_weights))
        self.print_fn('trainable in CSLModel: num params %d, num weights %d' % (
            len(self.params), sum([np.prod(param.size()) for param in self.params])))

    def build_submods(self):
        return {
            VISENC: t2vretrieval.encoders.video_encoder.VideoEncoder(self.config.subcfgs[VISENC]),
            TXTENC: t2vretrieval.encoders.text_encoder.TextEncoder(self.config.subcfgs[TXTENC]),
            VISDEC: SymbolGenerateModel(self.config.word_embed_path),
            TXTDEC: MemorySelectModel()
        }

    def forward_video_embed(self, batch_data):
        vid_fts = torch.FloatTensor(batch_data['attn_fts']).to(self.device)
        vid_lens = torch.LongTensor(batch_data['attn_lens']).to(self.device)
        # (batch, max_vis_len, dim_embed)
        vid_sent_embeds, frame_embeds = self.submods[VISENC](vid_fts, vid_lens)
        return {
            'vid_sent_embeds': vid_sent_embeds,
            'vid_lens': vid_lens,
            'frame_embeds': frame_embeds,
        }

    def forward_text_embed(self, batch_data):
        sent_ids = torch.LongTensor(batch_data['sent_ids']).to(self.device)
        sent_lens = torch.LongTensor(batch_data['sent_lens']).to(self.device)
        sent_embeds, sent_symbol_embeds = self.submods[TXTENC](sent_ids, sent_lens)
        return {
            'sent_embeds': sent_embeds, 'sent_lens': sent_lens,
            'sent_symbol_embeds': sent_symbol_embeds
        }

    def generate_scores(self, **kwargs):
        latent_scores = cosine_sim(kwargs['vid_sent_embeds'], kwargs['gen_vid_embeds'])
        symbol_scores = cosine_sim(kwargs['gen_sent_embeds'], kwargs['sent_embeds'])

        return latent_scores, symbol_scores



    def forward_symbolgenerate(self, batch_data, enc_outs, mode='train'):
        fc_feats = enc_outs['vid_sent_embeds']
        if mode == 'train':
            batch_size = fc_feats.size(0)
            caption_label = torch.LongTensor(batch_data['caption_label']).to(self.device)
            caption_mask = torch.FloatTensor(batch_data['caption_mask']).to(self.device)
            seq_probs, sent_embeds = self.submods[VISDEC](fc_feats, caption_label, mode)
            loss = self.crit(seq_probs, caption_label[:, 1:], caption_mask[:, 1:]) / batch_size
            return loss, sent_embeds
        else:
            fc_feats = enc_outs['vid_sent_embeds']
            seq_preds, sent_embeds = self.submods[VISDEC](fc_feats, None, mode)
            return seq_preds, sent_embeds

    def forward_memselect(self, enc_outs):
        fc_feats = enc_outs['sent_embeds']
        video_pattern = [m.weight.data.detach() for m in self.submods[VISENC].ft_attn_list]
        video_gen_embeds = self.submods[TXTDEC](fc_feats, video_pattern)
        return video_gen_embeds




    def forward_loss(self, batch_data, step=None, ganmode='D'):
        self.step = step
        enc_outs = self.forward_video_embed(batch_data)
        cap_enc_outs = self.forward_text_embed(batch_data)
        enc_outs.update(cap_enc_outs)
        generate_loss, gen_sent_embeds = self.forward_symbolgenerate(batch_data, enc_outs)
        enc_outs.update({'gen_sent_embeds': gen_sent_embeds})
        gen_vid_embeds = self.forward_memselect(enc_outs)
        enc_outs.update({'gen_vid_embeds': gen_vid_embeds})
        latent_scores, symbol_scores = self.generate_scores(**enc_outs)


        latent_loss = self.criterion(latent_scores)
        symbol_loss = self.criterion(symbol_scores)
        scores = (latent_scores + symbol_scores) / 2
        fused_loss = self.criterion(scores)


        loss = 1.00 * fused_loss + 1.00 * generate_loss


        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
            neg_scores = scores.masked_fill(torch.eye(len(scores), dtype=torch.bool).to(self.device), -1e10)
            self.print_fn('\tstep %d: pos mean scores %.2f, hard neg mean scores i2t %.2f, t2i %.2f' % (
                step, torch.mean(torch.diag(scores)), torch.mean(torch.max(neg_scores, 1)[0]),
                torch.mean(torch.max(neg_scores, 0)[0])))
            self.print_fn('\tstep %d: fused_loss %.4f, latent_loss %.4f, symbol_loss %.4f,  generate_loss %.4f' % (
                step, fused_loss.data.item(), latent_loss.data.item(), symbol_loss.data.item(),
                generate_loss.data.item()))

        return loss


    def evaluate_scores(self, tst_reader):
        K = 2
        vid_names, all_scores = [], [[] for _ in range(K)]
        cap_names = tst_reader.dataset.captions
        for vid_data in tst_reader:
            vid_names.extend(vid_data['names'])
            vid_enc_outs = self.forward_video_embed(vid_data)
            _, gen_sent_embeds = self.forward_symbolgenerate(vid_data, vid_enc_outs, 'test')
            vid_enc_outs.update({'gen_sent_embeds': gen_sent_embeds})
            for k in range(K):
                all_scores[k].append([])
            for cap_data in tst_reader.dataset.iterate_over_captions(self.config.tst_batch_size):
                cap_enc_outs = self.forward_text_embed(cap_data)
                cap_enc_outs.update(vid_enc_outs)
                gen_vid_embeds = self.forward_memselect(cap_enc_outs)
                cap_enc_outs.update({'gen_vid_embeds': gen_vid_embeds})
                indv_scores = self.generate_scores(**cap_enc_outs)
                for k in range(K):
                    all_scores[k][-1].append(indv_scores[k].data.cpu().numpy())
            for k in range(K):
                all_scores[k][-1] = np.concatenate(all_scores[k][-1], axis=1)
        for k in range(K):
            all_scores[k] = np.concatenate(all_scores[k], axis=0)
        all_scores = np.array(all_scores)

        return vid_names, cap_names, all_scores


    def evaluate(self, tst_reader, return_outs=False):
        with torch.no_grad():
            vid_names, cap_names, scores = self.evaluate_scores(tst_reader)
        i2t_gts = []
        for vid_name in vid_names:
            i2t_gts.append([])
            for i, cap_name in enumerate(cap_names):
                if cap_name in tst_reader.dataset.ref_captions[vid_name]:
                    i2t_gts[-1].append(i)

        t2i_gts = {}
        for i, t_gts in enumerate(i2t_gts):
            for t_gt in t_gts:
                t2i_gts.setdefault(t_gt, [])
                t2i_gts[t_gt].append(i)

        fused_scores = np.mean(scores, 0)

        metrics = self.calculate_metrics(fused_scores, i2t_gts, t2i_gts)

        metrics_latent = self.calculate_metrics(scores[0], i2t_gts, t2i_gts)
        metrics_symbol = self.calculate_metrics(scores[1], i2t_gts, t2i_gts)
        self.pretty_print_metrics('val for latent space', metrics_latent)
        self.pretty_print_metrics('val for symbol space', metrics_symbol)


        if return_outs:
            outs = {
                'vid_names': vid_names,
                'cap_names': cap_names,
                'scores': scores,
            }
            return metrics, outs
        else:
            return metrics


    def build_optimizer(self):
        trn_params = []
        trn_param_ids = set()
        per_param_opts = []
        for key, submod in self.submods.items():
            if self.config.subcfgs[key].freeze:
                for param in submod.parameters():
                    param.requires_grad = False
            else:
                params = []
                for name, param in submod.named_parameters():
                    if param.requires_grad and id(param) not in trn_param_ids:
                        params.append(param)
                        trn_param_ids.add(id(param))
                        continue
                per_param_opts.append({
                    'params': params,
                    'lr': self.config.base_lr * self.config.subcfgs[key].lr_mult,
                    'weight_decay': self.config.subcfgs[key].weight_decay,
                })
                trn_params.extend(params)
        if len(trn_params) >  0:
            optimizer = optim.Adam(per_param_opts, lr=self.config.base_lr)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                          milestones=self.config.decay_boundarys,
                                                          gamma=self.config.decay_rate)
        else:
            optimizer, lr_scheduler = None, None
            print('no traiable parameters')
        return trn_params, optimizer, lr_scheduler

    def train_one_batch(self, batch_data, step):
        loss = self.forward_loss(batch_data, step=step)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.data.item()
        if step is not None and self.config.monitor_iter > 0 and step % self.config.monitor_iter == 0:
            self.print_fn('\tmodel trn step %d lr %.8f %s: %.4f' % (step, self.get_current_base_lr(), 'loss', loss_value))
        return {'loss': loss_value}


    def train_one_epoch(self, step, trn_reader, val_reader, model_dir, log_dir):
        self.train_start()
        avg_loss, n_batches = {}, {}
        for batch_data in trn_reader:
            loss = self.train_one_batch(batch_data, step)
            for loss_key, loss_value in loss.items():
                avg_loss.setdefault(loss_key, 0)
                n_batches.setdefault(loss_key, 0)
                avg_loss[loss_key] += loss_value
                n_batches[loss_key] += 1
            step += 1

            if self.config.save_iter > 0 and step % self.config.save_iter == 0:
                self.save_checkpoint(os.path.join(model_dir, 'step.%d.th' % step))

            if (self.config.save_iter > 0 and step % self.config.save_iter == 0) \
                    or (self.config.val_iter > 0 and step % self.config.val_iter == 0):
                metrics = self.validate(val_reader, step=step)
                with open(os.path.join(log_dir, 'val.step.%d.json' % step), 'w') as f:
                    json.dump(metrics, f, indent=2)
                self.pretty_print_metrics('\tval step %d' % step, metrics)
                self.train_start()
        for loss_key, loss_value in avg_loss.items():
            avg_loss[loss_key] = loss_value / n_batches[loss_key]
        return avg_loss, step


    def get_current_base_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def epoch_postprocess(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train(self, trn_reader, val_reader, model_dir, log_dir, resume_file=None):
        assert self.optimizer is not None

        if resume_file is not None:
            self.load_checkpoint(resume_file)

        # first validate
        metrics = self.validate(val_reader)
        self.pretty_print_metrics('init val', metrics)

        # training
        step = 0
        for epoch in range(self.config.num_epoch):
            avg_loss, step = self.train_one_epoch(
                step, trn_reader, val_reader, model_dir, log_dir)
            self.pretty_print_metrics('epoch (%d/%d) trn' % (epoch, self.config.num_epoch), avg_loss)
            self.epoch_postprocess()


            if self.config.save_per_epoch:
                self.save_checkpoint(os.path.join(model_dir, 'epoch.%d.th' % epoch))

            if self.config.val_per_epoch:
                metrics = self.validate(val_reader, step=step)
                with open(os.path.join(log_dir,
                                       'val.epoch.%d.step.%d.json' % (epoch, step)), 'w') as f:
                    json.dump(metrics, f, indent=2)
                self.pretty_print_metrics('epoch (%d/%d) val' % (epoch, self.config.num_epoch), metrics)
