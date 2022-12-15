import os 
import torch
import torch.nn as nn
from torchaudio import transforms as T
import torch.nn.functional as F
from torchinfo import summary
from augmentations import SpecAugment
from models import CNN6, CNN10, CNN14, Projector, LinearClassifier
from dataset import ICBHI
from utils import Normalize, Standardize
from losses import SupConLoss, SupConCELoss
from ce import train_ce
from hybrid import train_supconce
from scl import train_scl, linear_scl
from args import args

#判断是否产生了csv文件
if not(os.path.isfile(os.path.join(args.datapath, args.metadata))):
    raise(IOError(f"CSV file {args.metadata} does not exist in {args.datapath}"))

METHOD = args.method

DEFAULT_NUM_CLASSES = 4 #for cross entropy
DEFAULT_OUT_DIM = 128  #for ssl embedding space dimension
DEFAULT_NFFT = 1024
DEFAULT_NMELS = 64
DEFAULT_WIN_LENGTH = 1024
DEFAULT_HOP_LENGTH = 512
DEFAULT_FMIN = 50
DEFAULT_FMAX = 2000

# Model definition
if args.method == 'sl':
    embed_only = False
else:
    embed_only = True
    projector = Projector(name=args.backbone, out_dim=DEFAULT_OUT_DIM, device=args.device)
    classifier = LinearClassifier(name=args.backbone, num_classes=DEFAULT_NUM_CLASSES, device=args.device)
    
if args.backbone == 'cnn6':
    PATH_TO_WEIGHTS = os.path.join(args.weightspath, 'Cnn6_mAP=0.343.pth')
    model = CNN6(num_classes=DEFAULT_NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)
elif args.backbone == 'cnn10':
    PATH_TO_WEIGHTS = os.path.join(args.weightspath, 'Cnn10_mAP=0.380.pth')
    model = CNN10(num_classes=DEFAULT_NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)
elif args.backbone == 'cnn14':
    PATH_TO_WEIGHTS = os.path.join(args.weightspath, 'Cnn14_mAP=0.431.pth')
    model = CNN14(num_classes=DEFAULT_NUM_CLASSES, do_dropout=args.dropout, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)
s = summary(model, device=args.device)
nparams = s.trainable_params

# Spectrogram definition
'''
将音频信号转换为时频梅尔谱图,使用64梅尔滤波器组,1024的窗口大小和512的跳大小
最小和最大频率分别为50hz和2000hz,因为喘息声和噼啪声都在这个区间内
我们还将mel谱图从功率尺度转换到分贝尺度,并在(0,1)范围内对所有输入分别进行最小-最大归一化
然后使用训练集的均值和标准差对输入进行标准化。
然后再经过SpecAugment，帮助网络学习对频率和时间信息的部分丢失和时间方向上的变形具有鲁棒性的特征。
'''
melspec = T.MelSpectrogram(n_fft=DEFAULT_NFFT, n_mels=DEFAULT_NMELS, win_length=DEFAULT_WIN_LENGTH, hop_length=DEFAULT_HOP_LENGTH, f_min=DEFAULT_FMIN, f_max=DEFAULT_FMAX).to(args.device)
normalize = Normalize()
melspec = torch.nn.Sequential(melspec, normalize)

# Data transformations
specaug = SpecAugment(freq_mask=args.freqmask, time_mask=args.timemask, freq_stripes=args.freqstripes, time_stripes=args.timestripes).to(args.device)
standardize = Standardize(device=args.device)
train_transform = nn.Sequential(melspec, specaug, standardize)
val_transform = nn.Sequential(melspec, standardize)

'''
获取数据的dataloader
'''
# Dataset and dataloaders
train_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='train', device=args.device, samplerate=args.samplerate, pad_type=args.pad)
val_ds = ICBHI(data_path=args.datapath, metadatafile=args.metadata, duration=args.duration, split='test', device=args.device, samplerate=args.samplerate, pad_type=args.pad)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs, shuffle=False)

'''
根据损失函数选择优化器
sl为交叉熵损失
scl为有监督对比损失
hybrid为交叉熵损失和有监督对比损失结合

我们使用余弦退火作为学习速率计划，而不需要热重启
除了监督对比的第二阶段,我们使用0.1的学习速率在固定表示上训练线性分类器，而不需要调度
'''
### OPTIMISER
if METHOD == 'sl':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif METHOD == 'scl':
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.wd) 
    optimizer2 = torch.optim.Adam(classifier.parameters(), lr=args.lr2, weight_decay=args.wd)
elif METHOD == 'hybrid':
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6) 

if args.noweights:
    criterion_ce = nn.CrossEntropyLoss()  
else:
    weights = torch.tensor([2063, 1215, 501, 363], dtype=torch.float32) #N_COUNT, C_COUNT, W_COUNT, B_COUNT = 2063, 1215, 501, 363 for trainset
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    weights = weights.to(args.device)    
    criterion_ce = nn.CrossEntropyLoss(weight=weights)

if METHOD == 'sl':
    history = train_ce(model, train_loader, val_loader, train_transform, val_transform, criterion_ce, optimizer, args.epochs, scheduler)
    del model

elif METHOD == 'scl':
    criterion = SupConLoss(temperature=args.tau, device=args.device)
    ssl_train_losses, model, last_checkpoint = train_scl(model, projector, train_loader, train_transform, criterion, optimizer, scheduler, args.epochs)
    history = linear_scl(model, last_checkpoint ,classifier, train_loader, val_loader, val_transform, criterion_ce, optimizer2, args.epochs2)
    del model; del projector; del classifier
    
elif METHOD == 'hybrid':
    criterion = SupConCELoss(temperature=args.tau, weights=weights, alpha=args.alpha, device=args.device)
    history = train_supconce(model, projector, classifier, train_loader, val_loader, train_transform, val_transform, criterion, criterion_ce, optimizer, args.epochs, scheduler)
    del model; del projector; del classifier

del train_ds; del val_ds
del train_loader; del val_loader
