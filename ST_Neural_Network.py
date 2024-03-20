"Copyright (C) S.Takeuchi 2024 All Rights Reserved."

import torch
import torch.nn as nn

class AirNN(nn.Module):
    def __init__(self,nch=16,output_layer=100,conv_kernel_size=(1,3),padding_size=(0,1),pool='ave',pool_kernel_size=(1,2),batch_norm=True):
        super(AirNN,self).__init__()

        #活性化関数定義
        self.active_func_mid=nn.ReLU()#中間層
        self.active_func_out=nn.ReLU()#出力層

        #プーリング層定義
        if pool=='max':
            self.pool=nn.MaxPool2d(kernel_size=pool_kernel_size,stride=2)
        elif pool=='ave':
            self.pool=nn.AvgPool2d(kernel_size=pool_kernel_size,stride=2)

        self.TCB1=_EncoderBlock(1,nch,nch,conv_kernel_size,padding_size,batch_norm)
        self.TCB2=_EncoderBlock(nch,nch*2,nch*2,conv_kernel_size,padding_size,batch_norm)
        self.TCB3=_EncoderBlock(nch*2,nch*4,nch*4,conv_kernel_size,padding_size,batch_norm)

        self.fc1=nn.Linear(in_features=5760,out_features=3000)
        self.fc2=nn.Linear(in_features=3000,out_features=1000)
        self.fc3=nn.Linear(in_features=1000,out_features=output_layer)

    def forward(self,x):
        x=self.TCB1(x)
        x=self.pool(x)
        x1=x

        x=self.TCB2(x)
        x=self.pool(x)
        x2=x

        x=self.TCB3(x)

        x1=x1.view(x1.size()[0],-1)
        x2=x2.view(x2.size()[0],-1)
        xx=x.view(x.size()[0],-1)
        x=torch.cat((x1,x2,xx),dim=1)

        x=self.fc1(x)
        x=self.active_func_mid(x)
        x=self.fc2(x)
        x=self.active_func_mid(x)
        x=self.fc3(x)
        x=self.active_func_out(x)
        return x

class _EncoderBlock(nn.Module):
    def __init__(self,in_channels,middle_channels,out_channels,conv_kernel_size,padding_size,batch_norm):
        super(_EncoderBlock,self).__init__()
        #活性化関数定義
        self.active_func_mid=nn.ReLU()#中間層

        self._batch_norm=batch_norm

        #畳み込み層定義
        self.conv1=nn.Conv2d(in_channels,middle_channels,kernel_size=conv_kernel_size,stride=(1,1),padding=padding_size,padding_mode='replicate')
        self.bn1=nn.BatchNorm2d(middle_channels)
        self.conv2=nn.Conv2d(middle_channels,out_channels,kernel_size=conv_kernel_size,stride=(1,1),padding=padding_size,padding_mode='replicate')
        self.bn2=nn.BatchNorm2d(out_channels)

    def forward(self,x):
        if self._batch_norm:
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.active_func_mid(x)
            x=self.conv2(x)
            x=self.bn2(x)
            x=self.active_func_mid(x)
        else:
            x=self.conv1(x)
            x=self.active_func_mid(x)
            x=self.conv2(x)
            x=self.active_func_mid(x)
        return x