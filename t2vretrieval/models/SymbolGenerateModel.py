import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase

class SymbolGenerateModelConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()

class SymbolGenerateModel(nn.Module):
    def __init__(self, word_embed_path):
        """
        """
        super(SymbolGenerateModel, self).__init__()
        self.word_embed_path = word_embed_path
        self.sos_id = 0
        self.hidden_size = 1024
        self.seq_len = 30
        self.vocab_size = 10510 #10424 #6862 #10510 #10424 #10510 #30522 #10510
        self.word_dim = 300 #768 #300
        self.h0_embed = nn.Linear(self.hidden_size, self.hidden_size)
        self.c0_embed = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_embed = nn.Linear(self.word_dim, self.hidden_size)
        self.out_embed = nn.Linear(self.hidden_size, self.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.word_dim)

        self.lstm_cell = nn.LSTMCell(input_size=self.hidden_size+self.hidden_size, hidden_size=self.hidden_size)
        self.init_weights()
        self.ft_attn = nn.Linear(1024, 1)
        self.PFS = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=False), nn.Linear(1024, 1024), nn.LayerNorm((1024, )), nn.ReLU())

    def init_weights(self):
        state_dicts = torch.load(self.word_embed_path, map_location=lambda storage, loc: storage)
        embeds_param = state_dicts['text_encoder']['embedding.we.weight']
        new_stat_dicts = {'weight': embeds_param}
        self.embedding.load_state_dict(new_stat_dicts)

    def init_hidden(self, feature):
        return self.h0_embed(feature), self.c0_embed(feature)



    def forward(self, vid_feats, targets, mode='train', sample_max=True, p=0.0):
        """
        Args:
            vid_feats (Variable): video feats of shape [batch_size, seq_len, dim_vid]
            target_variable (None, optional): groung truth labels

        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        batch_size = vid_feats.size(0)
        probs = []
        preds = []
        hiddens = []
        probs_train = []
        if mode == 'train':
            h, c = self.init_hidden(vid_feats)
            targets_emb = self.embedding(targets)
            targets_emb = self.w_embed(targets_emb)
            for t in range(self.seq_len-1):
                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).to(vid_feats.device)
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    probs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long().to(vid_feats.device)

                preds.append(it.view(-1, 1))
                xt = self.embedding(it)
                xt = self.w_embed(xt)
                tf_rate = np.random.rand()
                current_words = targets_emb[:, t, :]
                if tf_rate < p:
                    input = torch.cat([current_words, vid_feats], dim=1)
                else:
                    input = torch.cat([xt, vid_feats], dim=1)

                h, c = self.lstm_cell(input, (h, c))
                hiddens.append(h.unsqueeze(dim=1))
                prob = self.out_embed(h)
                logprobs = F.log_softmax(prob, dim=1)
                probs_train.append(logprobs.unsqueeze(1))
            probs_train = torch.cat(probs_train, 1)
            hiddens = torch.cat(hiddens, dim=1)
            sent_embeds = torch.mean(hiddens, dim=1)

            return probs_train, sent_embeds
        else:
            h, c = self.init_hidden(vid_feats)
            for t in range(self.seq_len):
                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).to(vid_feats.device)
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    probs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long().to(vid_feats.device)

                preds.append(it.view(-1, 1))

                xt = self.embedding(it)
                xt = self.w_embed(xt)
                input = torch.cat([xt, vid_feats], dim=1)
                h, c = self.lstm_cell(input, (h, c))
                hiddens.append(h.unsqueeze(dim=1))
                prob = self.out_embed(h)
                logprobs = F.log_softmax(prob, dim=1)
            preds = torch.cat(preds, 1)
            hiddens = torch.cat(hiddens[:-1], dim=1)
            sent_embeds = torch.mean(hiddens, dim=1)

            return preds[:, 1:], sent_embeds

