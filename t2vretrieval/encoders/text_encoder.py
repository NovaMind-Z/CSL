import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase
import framework.ops

import t2vretrieval.encoders.sentence


class TextEncoderConfig(t2vretrieval.encoders.sentence.SentEncoderConfig):
    def __init__(self):
        super().__init__()

class TextEncoder(t2vretrieval.encoders.sentence.SentAttnEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.concept_embeds = nn.Linear(1024, 1024, bias=True)

    def forward(self, sent_ids, sent_lens):
        '''
        Args:
          sent_ids: (batch, max_sent_len)
          sent_lens: (batch, )
        '''
        # (batch, max_sent_len, embed_size)
        word_embeds, word_attn_scores = super().forward(sent_ids, sent_lens, return_dense=True)
        # sent_embeds = torch.sum(word_embeds * word_attn_scores.unsqueeze(2), 1)
        sent_embeds = torch.mean(word_embeds, dim=1)
        sent_concept_embeds = self.concept_embeds(sent_embeds)

        return sent_embeds, sent_concept_embeds

