from torch import nn
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np
from efficientunet import get_efficientunet_b0

class LinearPatchProjection(nn.Module):
    def __init__(self, device, batch_size, out_dim=768, img_size=224, patch_size=16, channel_size=3):
        super(LinearPatchProjection, self).__init__()
        self.device = device
        self.b = batch_size
        self.p = patch_size
        self.c = channel_size
        self.out_dim = out_dim
        self.n = img_size ** 2 // (patch_size ** 2)
        
        self.projection = nn.Linear(in_features=self.p**2 * self.c, out_features=self.out_dim)
        
    def forward(self,x):
        x = x.view(-1, self.n, (self.p**2)*self.c)
        x_p = self.projection(x)
        x_cls = nn.Parameter(torch.randn(x_p.size(0), 1, self.out_dim), requires_grad=True).to(self.device)
        x_pos = nn.Parameter(torch.randn(x_p.size(0), self.n+1, self.out_dim), requires_grad=True).to(self.device)
        x_p = torch.cat((x_cls, x_p), dim=1)
        x = torch.add(x_p, x_pos)

        return x
    
class SelfAttention(nn.Module):
    def __init__(self, out_dim=768, d=12):
        super(SelfAttention, self).__init__()
        self.out_dim = out_dim
        self.norm_scale = out_dim // d
        self.Wq = nn.Linear(in_features=out_dim, out_features=out_dim // d)
        self.Wk = nn.Linear(in_features=out_dim, out_features=out_dim // d)
        self.Wv = nn.Linear(in_features=out_dim, out_features=out_dim // d)
        self.soft = nn.Softmax(dim=-1)
        
    def forward(self,x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        qk = torch.div(torch.matmul(q, torch.transpose(k, 1, 2)), self.norm_scale ** 0.5)
        qk = self.soft(qk)
        qkv = torch.matmul(qk, v)
        return qkv
    
class MultiHeadAttention(nn.Module):
    def __init__(self, out_dim=768, h=12):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.SA = nn.ModuleList([SelfAttention(out_dim, h) for _ in range(h)])
        self.linear = nn.Linear(in_features=out_dim, out_features=out_dim)
        
    def forward(self,x):
        for i in range(self.h):
            if i == 0:
                x_cat = self.SA[i](x)
            else:
                x_cat = torch.cat((x_cat, self.SA[i](x)), dim=-1)
        x = self.linear(x_cat)
        return x
    
class Encoder(nn.Module):
    def __init__(self, out_dim=768, h=12):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(out_dim)
        self.act1 = nn.GELU()
        self.mha = MultiHeadAttention(out_dim, h=h)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act2 = nn.GELU()
        self.linear = nn.Linear(in_features=out_dim, out_features=out_dim)
        
    def forward(self, x):
        x_norm = self.norm1(x)
        x_norm = self.act1(x_norm)
        x_norm = self.mha(x_norm)
        x = torch.add(x_norm, x)
        x_norm = self.norm2(x)
        x_norm = self.act2(x_norm)
        x_norm = self.linear(x_norm)
        x = torch.add(x_norm, x)
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, device, L=12, out_dim=768, h=12, ML=3096, num_classes=8, img_size=224, patch_size=16, channel_size=3, batch_size=16):
        super(VisionTransformer, self).__init__()
        self.batch_size = batch_size
        self.embedding_my = LinearPatchProjection(device, self.batch_size, out_dim, img_size, patch_size, channel_size)
        self.transencoder = nn.Sequential(*[Encoder(out_dim, h) for _ in range(L)])
        self.flatten = nn.Flatten()
        self.mlphead = nn.Sequential(nn.Linear(((img_size // patch_size) ** 2 + 1) * out_dim, num_classes))
        self.soft = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.embedding_my(x)
        x = self.transencoder(x)
        x = self.flatten(x)
        x = self.mlphead(x)
        x = self.soft(x)
        return x

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
    def forward(self, input_):
        x0_0 = self.conv0_0(input_)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output
    
class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        
        nb_filter = [32, 64, 128, 256, 512]
        
        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        
        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        
        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        
        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        
        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        
    def forward(self, input_):
        x0_0 = self.conv0_0(input_)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(x0_4)
        return output
    
class UNetTransformer(nn.Module):
    def __init__(self, device):
        super(UNetTransformer, self).__init__()
        u_model = UNet(3).cuda()
        u_model.load_state_dict(torch.load(f'./Results/Unet.pt'))
        self.unet = u_model
        self.ViT = VisionTransformer(device)
    
    def forward(self,x):
        x = self.unet(x)
        x = self.ViT(x)
        return x

class UNetPPTransformer(nn.Module):
    def __init__(self, device):
        super(UNetPPTransformer, self).__init__()
        u_model = NestedUNet(3).cuda()
        u_model.load_state_dict(torch.load(f'./Results/Unetpp.pt'))
        self.unetpp = u_model
        self.ViT = VisionTransformer(device)
    
    def forward(self,x):
        x = self.unetpp(x)
        x = self.ViT(x)
        return x
    
class EfficientUNetTransformer(nn.Module):
    def __init__(self, device):
        super(EfficientUNetTransformer, self).__init__()
        u_model = get_efficientunet_b0(out_channels=3, concat_input=True, pretrained=False).cuda()
        u_model.load_state_dict(torch.load(f'./Results/EffUnet.pt'))
        self.unet = u_model
        self.ViT = VisionTransformer(device)
    
    def forward(self,x):
        x = self.unet(x)
        x = self.ViT(x)
        return x