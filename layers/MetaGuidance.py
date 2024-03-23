import torch
import torch.nn as nn
import torch.fft
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


class GuidanceEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(GuidanceEmbedding, self).__init__()
        self.d_model = d_model
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, guidance):
        x = self.tokenConv((x * guidance).permute(0, 2, 1)).transpose(1, 2)
        return x

class Projector_Multi(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims = [16, 16], hidden_layers = 2, kernel_size=3):
        super(Projector_Multi, self).__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2*enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], enc_in, bias=False), nn.Tanh()]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, period):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, period.unsqueeze(1)], dim=1)
        x = x.view(batch_size, -1)
        y = self.backbone(x)
        return y


class Projector_Ratio(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims = [16, 16], hidden_layers = 2, kernel_size=3):
        super(Projector_Ratio, self).__init__()
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv_x = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.series_conv_trend = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)
        self.series_conv_period = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(3 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], enc_in, bias=False), nn.Sigmoid()]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, trend, period):
        batch_size = x.shape[0]
        x = self.series_conv_x(x)
        trend = self.series_conv_trend(trend)
        period = self.series_conv_period(period)
        x = torch.cat([x, trend, period], dim=1)
        x = x.view(batch_size, -1)
        y = self.backbone(x)
        return y



class GuidanceGenerator(nn.Module):
    def __init__(self, d_feature, seq_len, k,r):
        super(GuidanceGenerator, self).__init__()
        self.num_features = d_feature
        self.seq_len = seq_len
        self.eps = 1e-5
        self._init_params()
        self.k = k
        self.r = r

    

    def _init_params(self):
        self.period_multi_learner = Projector_Multi(enc_in=self.num_features, seq_len=self.seq_len)
        self.trend_multi_learner = Projector_Multi(enc_in=self.num_features, seq_len=self.seq_len)
        self.trend_period_ratio_learner = Projector_Ratio(enc_in=self.num_features, seq_len=self.seq_len)

    def forward(self, x, mode:str, mask, x_interpolate):
        if mode == 'norm':
            x = self._normalize(x, mask, x_interpolate)
        
        elif mode == 'denorm':
            x = self._denormalize(x)
        
        else: raise NotImplementedError
        return x    


    def SelectRange(self, x, minn, maxx, pdim):
        ind = (((x[:, pdim] < minn) | (x[:, pdim] >= maxx))).nonzero()
        ind = torch.unique(ind)
        ind_r = torch.arange(x.shape[0]).to(x.device)
        ind_r = ind_r[~torch.isin(ind_r, ind)]
        result_tensor = torch.index_select(x, dim=0, index=ind_r)
        return result_tensor


    def NormalDistribution(self, i, multi, sigma):
        return multi * np.exp(-1 * i*i/(2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    

    def get_guidance_general(self, x, mask, period, multi):
        sigma = 1
        guidance = torch.ones_like(x).to(x.device)
        pos = (mask == 0).nonzero()
        for i in range(-self.r, self.r+1):
            if i==0: continue
            pos_py = pos.clone()
            pos_py[:, 1] += i * period[pos_py[:, 0], pos_py[:, 2]].long()
            pos_py = self.SelectRange(pos_py, 0, x.shape[1], 1)
            guidance[pos_py[:, 0], pos_py[:, 1], pos_py[:, 2]] += self.NormalDistribution(i, multi[pos_py[:, 0], pos_py[:, 2]], sigma)

        return guidance
    
    
    def _get_guidance(self, x, mask, x_interpolate):
        # ns guidance
        x_d = x.clone().detach()
        x_d = torch.nn.functional.normalize(x_d, dim=1)
        ones = torch.ones(x.shape[0], x.shape[2], dtype=torch.int64).to(x.device)
        multi_trend = self.trend_multi_learner(x_d, ones).exp()
        guidance_trend = self.get_guidance_general(x, mask, ones, multi_trend)

        # per guidance
        seq_len = x.shape[1]
        N_2 = seq_len // 2

        x_fft = torch.fft.fft(x_interpolate, dim=1).to(x.device)
        x_fft = torch.abs(x_fft)[:, :N_2, :]
        frequency = torch.fft.fftfreq(seq_len)[:N_2].to(x.device)

        k=self.k
        amplitudes, index = torch.topk(x_fft, k, dim=1)
        index = index.to(x.device)
        amplitudeSum = torch.sum(amplitudes, dim=1) + self.eps
        x_int = x_interpolate.clone().detach()
        x_int = torch.nn.functional.normalize(x_int, dim=1)
        guidance_period = torch.zeros(x.shape).to(x.device)

        for i in range(k):
            max_period = torch.reciprocal(frequency[index[:,i,:]]).to(x.device) # B, N
            max_period[max_period >= self.seq_len] = 1
            max_period = max_period.round()
            rate = (amplitudes[:,i,:] / amplitudeSum).unsqueeze(1)

            multi_period = self.period_multi_learner(x_int, max_period).exp()
            sub_guidance_period = self.get_guidance_general(x, mask, max_period, multi_period)
            guidance_period += sub_guidance_period*rate

        ratio = self.trend_period_ratio_learner(x_int, guidance_trend, guidance_period).unsqueeze(1)
        guidance = guidance_trend * ratio + guidance_period * (1-ratio)

        return guidance * mask


    def _normalize(self, x, mask, x_interpolate):
        guidance = self._get_guidance(x, mask, x_interpolate)

        x_d = x * guidance
        cnt = torch.sum(guidance, dim=1)
        cnt[cnt == 0] = 1

        mean = torch.sum(x_d, dim=1) / cnt
        mean = mean.unsqueeze(1)
        self.mean = mean
        
        x = x - self.mean
        x = x.masked_fill(mask == 0, 0)
        x_d = x

        stdev = torch.sqrt(torch.sum(x_d * x_d * guidance, dim=1) / cnt + self.eps)
        stdev = stdev.unsqueeze(1)
        self.stdev = stdev
        x = x / self.stdev

        return x, guidance * mask

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean

        return x