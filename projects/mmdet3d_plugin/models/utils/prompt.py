from turtle import forward
import torch
import torch.nn as nn


class PadPrompter(nn.Module):
    def __init__(self, c, w, h, eta=0.3):
        super(PadPrompter, self).__init__()
        pad_w = int(eta * w) + 1
        pad_h = int(eta * h) + 1
        feat_h = h
        feat_w = w

        self.base_h = feat_h - 2*pad_h
        self.base_w = feat_w - 2*pad_w
        self.base_c = c

        self.pad_up = nn.Parameter(torch.randn(
            [1, self.base_c, pad_h, feat_w]))
        self.pad_down = nn.Parameter(torch.randn(
            [1, self.base_c, pad_h, feat_w]))
        self.pad_left = nn.Parameter(torch.randn(
            [1, self.base_c, feat_h - 2*pad_h, pad_w]))
        self.pad_right = nn.Parameter(torch.randn(
            [1, self.base_c, feat_h - 2*pad_h, pad_w]))

    def forward(self, x):
        bs, t, n, c, hf, wf = x.size()
        x = x.reshape(bs*t*n, c, hf, wf)
        base = torch.zeros(1, c, self.base_h, self.base_w).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return (x + prompt).reshape(bs, t, n, c, hf, wf)
