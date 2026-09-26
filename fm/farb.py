# -*- coding: utf-8 -*-
   
                                                                             

                 
                                                                                
                                                                           
                                                                              
                                                      
  
                                                                             
                                                                        
  
                                         
                                               
                                              

                   
                                                                                    
                                                               
                                                             
                                                                         
   

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def timestep_embedding(t, dim=128):
                                                                   
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class FreqChannelAttention(nn.Module):
                                                                    
    
                                         
                                                          
                                       
                                          
    
                                               
       
    def __init__(self, channels, t_dim=128, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels + t_dim, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x, t_emb):
           
             
                                       
                                                
                
                                                
           
        B, C, H, W = x.shape
                                         
        gap = x.mean(dim=[2, 3])
                                               
        feat = torch.cat([gap, t_emb], dim=-1)
                                  
        scale = self.fc(feat).view(B, C, 1, 1)
        return x * scale


class FARB(nn.Module):
                                      
    
                                                                                    
                                                                           
                                                             
    
                 
                                                                                                
    
                                                      
       
    def __init__(self, channels, t_dim=128, reduction=4):
        super().__init__()
        self.channels = channels
        self.t_dim = t_dim
        
                                                         
                                          
        self.low_proj = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
                                        
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )
        
                                                         
        self.low_attn = FreqChannelAttention(channels, t_dim, reduction)
        self.high_attn = FreqChannelAttention(channels, t_dim, reduction)
        
               
        self.merge = nn.Conv2d(channels * 2, channels, 1)
        
                                                                            
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x, t):
           
             
                                                  
                                                           
                
                                                          
           
        B, C, H, W = x.shape
        
                                         
        t_norm = t / 999.0 if t.max() > 1.0 else t
        t_emb = timestep_embedding(t_norm, self.t_dim)              
        
                                 
        x_low = self.low_proj(x)                                            
        x_low_up = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
        x_high = self.high_proj(x) - x_low_up                                    
        
                                        
        x_low_att = self.low_attn(x_low_up, t_emb)                     
        x_high_att = self.high_attn(x_high, t_emb)                     
        
               
        merged = self.merge(torch.cat([x_low_att, x_high_att], dim=1))                
        
                                                  
        return x + self.scale * merged
