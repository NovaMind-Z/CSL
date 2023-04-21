import os
import json
import numpy as np
import h5py
import collections
import torch
from t2vretrieval.readers.utils import BigFile, read_dict

import t2vretrieval.readers.mpdata
BOS, EOS, UNK = 0, 1, 2
# BOS, EOS, UNK = 1, 2, 3

# get image id from caption id
def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    if vid_id.endswith('.jpg') or vid_id.endswith('.mp4'):
        vid_id = vid_id[:-4]
    return vid_id

class CSLDataset(t2vretrieval.readers.mpdata.MPDataset):
    def __init__(self, split, config, max_words_in_sent, max_attn_len=20, is_train=False, _logger=None):
        if _logger is None:
            self.print_fn = print
        else:
            self.print_fn = _logger.info

        self.cap_ids = []
        self.video_ids = []
        self.captions = []
        self.ref_captions = {}
        self.max_words_in_sent = max_words_in_sent
        self.is_train = is_train
        self.max_attn_len = max_attn_len
        self.cap_file = config.cap_root + split + '.caption.txt'

        self.resnext_ft_flies = config.visual_root

        self.visual_feats = BigFile(self.resnext_ft_flies)
        self.video2frames = read_dict(self.resnext_ft_flies + 'video2frames.txt')

        self.word2int = json.load(open(config.word2int_file))

        with open(self.cap_file, 'r') as cap_reader:
            id = 0
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                # if caption in self.captions:
                #     continue
                video_id = getVideoId(cap_id)
                if video_id in self.ref_captions:
                    self.ref_captions[video_id].append(caption)
                else:
                    self.ref_captions[video_id] = [caption]
                self.captions.append(caption)
                self.cap_ids.append(id)
                self.video_ids.append(video_id)
                id += 1
        if not self.is_train:
            self.video_ids = list(set(self.video_ids))
        self.num_pairs = len(self.video_ids)

        self.print_fn('num_videos %d' % len(self.video_ids))
        self.print_fn('captions size %d' % len(self.cap_ids))

    def __len__(self):
        return self.num_pairs


    def load_resnext_ft_by_name(self, video_id):
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feats.read_one(frame_id))
        frames_tensor = np.array(frame_vecs)

        return frames_tensor


    def pad_or_trim_feature(self, attn_ft, max_attn_len, trim_type='top'):
        seq_len, dim_ft = attn_ft.shape
        attn_len = min(seq_len, max_attn_len)

        # pad
        if seq_len < max_attn_len:
            new_ft = np.zeros((max_attn_len, dim_ft), np.float32)
            new_ft[:seq_len] = attn_ft
        # trim
        else:
            if trim_type == 'top':
                new_ft = attn_ft[:max_attn_len]
            elif trim_type == 'select':
                idxs = np.round(np.linspace(0, seq_len - 1, max_attn_len)).astype(np.int32)
                new_ft = attn_ft[idxs]
        return new_ft, attn_len

    def process_sent(self, sent, max_words):
        tokens = [self.word2int.get(w, UNK) for w in sent.split()]
        # # add BOS, EOS?
        # tokens = [BOS] + tokens + [EOS]
        tokens = tokens[:max_words]
        tokens_len = len(tokens)
        tokens = np.array(tokens + [EOS] * (max_words - tokens_len))
        return tokens, tokens_len


    def get_caption_outs(self, out, sent):
        sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
        mask = np.zeros(self.max_words_in_sent)
        gts = np.zeros((self.max_words_in_sent))
        caption = sent.split()
        cap_caption = ['<BOS>'] + caption + ['<EOS>']
        if len(cap_caption) > self.max_words_in_sent - 1:
            cap_caption = cap_caption[:self.max_words_in_sent]
            cap_caption[-1] = '<EOS>'
        for j, w in enumerate(cap_caption):
            gts[j] = self.word2int.get(w, UNK)

        non_zero = gts.nonzero()
        mask[:int(non_zero[0][-1])+1] = 1

        out['sent_ids'] = sent_ids
        out['sent_lens'] = sent_len
        out['caption_label'] = gts
        out['caption_mask'] = mask
        out['caps_gt'] = cap_caption[1:-1]


        return out

    def __getitem__(self, idx):
        out = {}
        if self.is_train:
            video_idx, cap_idx = self.video_ids[idx], self.cap_ids[idx]
            sent = self.captions[cap_idx]
            out = self.get_caption_outs(out, sent)
        else:
            video_idx = self.video_ids[idx]

        attn_fts = self.load_resnext_ft_by_name(video_idx)
        attn_fts, attn_len = self.pad_or_trim_feature(attn_fts, self.max_attn_len, trim_type='select')

        out['names'] = video_idx
        out['attn_fts'] = attn_fts
        out['attn_lens'] = attn_len
        return out

    def iterate_over_captions(self, batch_size):
        # the sentence order is the same as self.captions
        for s in range(0, len(self.captions), batch_size):
            e = s + batch_size
            data = []
            for sent in self.captions[s: e]:
                out = self.get_caption_outs({}, sent)
                data.append(out)
            outs = collate_graph_fn(data)
            yield outs

def collate_graph_fn(data):
    outs = {}
    for key in ['names', 'attn_fts', 'attn_lens', 'sent_ids', 'sent_lens', 'caption_label', 'caption_mask', 'caps_gt', 'tokens', 'segments', 'input_masks']:
        if key in data[0]:
            outs[key] = [x[key] for x in data]

    # reduce attn_lens
    if 'attn_fts' in outs:
        max_len = np.max(outs['attn_lens'])
        outs['attn_fts'] = np.stack(outs['attn_fts'], 0)[:, :max_len]

    # reduce attn_lens
    if 'vid_tags' in outs:
        outs['vid_tags'] = np.stack(outs['vid_tags'], 0)

    # reduce caption_ids lens
    if 'sent_lens' in outs:
        max_cap_len = np.max(outs['sent_lens'])
        outs['sent_ids'] = np.array(outs['sent_ids'])[:, :max_cap_len]
    return outs