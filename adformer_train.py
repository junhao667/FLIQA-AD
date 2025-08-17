import pdb
import os 
import time
import random
import torch
import torch.nn as nn
import numpy as np
import datetime
from torch.utils.data import DataLoader
from models.adformer import ADFormer
from datasets.dataset import ImageTextContrastiveDataset,ZeroShotImageDataset
from datasets.dataset import ImageTextContrastiveCollator,ZeroShotImageCollator
from trainer.trainer_adformer import Trainer

# set random seed
seed = 40
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_config = {
    'batch_size':8,
    'num_epochs': 200,
    'warmup': 0.1,
    'lr':2e-6,
    'weight_decay': 1e-4,
    'eval_batch_size': 8,
    'eval_steps': 1000,
    'save_steps': 1000,
}

train_datalist = [
    'ADNI_train_mmse',
    'ADNI-AD'
]

val_datalist = [
    'ADNI-val'
    # 'OAS-MMSE' 
]

traindata = ImageTextContrastiveDataset(datalist=train_datalist)
train_collate_fn = ImageTextContrastiveCollator()
trainloader = DataLoader(traindata,
    batch_size=train_config['batch_size'],
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
    )

val_data = ZeroShotImageDataset(datalist=val_datalist)
val_collate_fn = ZeroShotImageCollator()
valloader = DataLoader(val_data,
    batch_size=train_config['eval_batch_size'],
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True
    )



model = ADFormer(
                # vit_model = 'clip_L', if you want to use 2D gray CLIP model
                vit_model= 'eva_clip_g'
                img_size = 126,
                freeze_vit =False,
                train_conv1=False,
                patch_size=18,
                t5_model="google/flan-t5-xl",
                bert_type='emilyalsentzer/Bio_ClinicalBERT',
                qformer_text_input=True
                )
print(train_config['lr'],train_config['batch_size'],train_config['num_epochs'])

model = model.cuda()
model_save_path = './checkpoints'
plot_loss_path =  './plot/loss'
plot_eval_path =  './plot/val'
regression_path = './plot/val/regression_val_Res'
trainer = Trainer()
trainer.train(
    model,
    trainloader,
    valloader,
    epochs=train_config['num_epochs'],
    warmup_ratio=train_config['warmup'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    weight_decay=train_config['weight_decay'],
    use_amp=True,
    plot_loss_path=plot_loss_path,
    plot_eval_path=plot_eval_path,
    val_regression_path=regression_path,
    )


   