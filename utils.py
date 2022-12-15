import torch
import torchaudio
from torchaudio import transforms as T

'''
用于Spectrogram definition和Data transformations
'''

#从功率尺度转换为分贝尺度，然后再归一化
class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #转换为分贝尺度
        self.power_to_db = T.AmplitudeToDB()
    def forward(self, x):
        x = self.power_to_db(x)
        return (x-x.min())/(x.max()-x.min())
        

#利用训练集的参数进行标准化
class Standardize(torch.nn.Module):
    def __init__(self, mean=0.3690, std=0.0255, device='cpu'): #official split mean & std
        super().__init__()
        self.mean = mean
        self.std = std
        self.device = device
    def forward(self, x):
        return (x-self.mean) / self.std