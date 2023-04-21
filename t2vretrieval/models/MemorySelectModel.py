import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.configbase


class MemorySelectModelConfig(framework.configbase.ModuleConfig):
  def __init__(self):
    super().__init__()

class FeatureSelector(nn.Module):
    def __init__(self, hidden_size):
        super(FeatureSelector, self).__init__()
        self.hidden_size = hidden_size
        self.PFS = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=False), nn.Linear(1024, 1024), nn.LayerNorm((1024,)), nn.ReLU())

    def forward(self, cap_feats, video_patterns):
        """
        :param cap_feats: [bsz, dim]
        :param video_pattern:  [1, dim]
        :return:  [bsz, dim]
        """
        masks = []
        for video_pattern in video_patterns:
            # feature select part
            sim = torch.abs(cap_feats - video_pattern)
            mask = self.PFS(sim)
            masks.append(mask.unsqueeze(dim=1))
        masks = torch.cat(masks, dim=1)
        mask_mean = torch.max(masks, dim=1)[0]
        refined_embeds = mask_mean * cap_feats

        # # feature select part naive
        # sim = cap_feats * video_pattern
        # mask = self.PFS(sim)
        # refined_embeds = mask * cap_feats

        return refined_embeds



class MemorySelectModel(nn.Module):
    def __init__(self):
        """

        """
        super(MemorySelectModel, self).__init__()
        self.seq_len = 20
        self.hidden_size = 1024
        self.video_size = 4096
        self.GNet = FeatureSelector(self.hidden_size)

    def forward(self, cap_feats, video_parttern):
        """
        """
        gen_vid_embeds = self.GNet(cap_feats, video_parttern)

        return gen_vid_embeds