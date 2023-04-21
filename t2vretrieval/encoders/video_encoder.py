import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.configbase

class VideoEncoderConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()
    self.dim_fts = 4096
    self.dim_embed = 1024
    self.dropout = 0

    self.num_memory = 9
    self.share_enc = False

class VideoEncoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    input_size = self.config.dim_fts
    self.num_memory = self.config.num_memory
    self.dropout = nn.Dropout(self.config.dropout)

    self.ft_embeds = nn.Linear(input_size, self.config.dim_embed, bias=True)


    self.ft_attn_list = nn.ModuleList([
      nn.Linear(self.config.dim_embed, 1, bias=True) for _ in range(self.num_memory)
    ])

  def forward(self, inputs, input_lens):
    '''
    Args:
      inputs: (batch, max_seq_len, dim_fts)
    Return:
      sent_embeds: (batch, dim_embed)
    '''
    embeds = []
    embeds.append(self.dropout(self.ft_embeds(inputs)))

    attn_scores_list = []
    for i in range(self.num_memory):
      attn_scores = self.ft_attn_list[i](embeds[0]).squeeze(2)  # (batch, max_seq_len)
      input_pad_masks = framework.ops.sequence_mask(input_lens, max_len=attn_scores.size(1), inverse=True)
      attn_scores = attn_scores.masked_fill(input_pad_masks, -1e18)
      attn_scores = torch.softmax(attn_scores, dim=1)
      attn_scores_list.append(attn_scores.unsqueeze(1))

    attn_scores_list = torch.cat(attn_scores_list, dim=1)
    #attn_scores_max = torch.max(attn_scores_list, dim=1)[0]
    attn_scores_max = torch.mean(attn_scores_list, dim=1)
    sent_embeds = torch.sum(embeds[0] * attn_scores_max.unsqueeze(2), 1)

    return sent_embeds, embeds[0]
