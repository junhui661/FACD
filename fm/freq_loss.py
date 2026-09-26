# -*- coding: utf-8 -*-
   
                                              
                                                 
                                         
   

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


                                                              
                                                      
                                                              

class HaarDWT2D(nn.Module):
                                                           
    
                                                         
                                                        
                                              
                                          
                                                
       
    def __init__(self):
        super().__init__()
                                                     
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        
                                                
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)
    
    def forward(self, x):
           
             
                                  
                
                                                                        
           
        B, C, H, W = x.shape
                                                      
        x_flat = x.reshape(B * C, 1, H, W)
                                          
        out = F.conv2d(x_flat, self.filters, stride=2, padding=0)                      
        out = out.reshape(B, C, 4, H // 2, W // 2)
        
        return {
            'LL': out[:, :, 0],                     
            'LH': out[:, :, 1],
            'HL': out[:, :, 2],
            'HH': out[:, :, 3],
        }


class MultiLevelDWT(nn.Module):
                                     
    
                               
                                        
                                                            
    
                                     
                                    
                                                
                                                    
       
    def __init__(self, levels=2):
        super().__init__()
        self.levels = levels
        self.dwt = HaarDWT2D()
    
    def forward(self, x):
                                                            
                 
        bands_1 = self.dwt(x)
        
        if self.levels == 1:
            return {
                'low': bands_1['LL'],
                'mid': torch.cat([bands_1['LH'], bands_1['HL']], dim=1),
                'high': bands_1['HH'],
            }
        
                                    
        bands_2 = self.dwt(bands_1['LL'])
        
        return {
            'low': bands_2['LL'],                                        
            'mid': torch.cat([                                      
                bands_2['LH'], bands_2['HL'], bands_2['HH']
            ], dim=1),
            'high': torch.cat([                                     
                bands_1['LH'], bands_1['HL'], bands_1['HH']
            ], dim=1),
        }


                                                              
                            
                                                              

def sinusoidal_embedding(t, dim=64):
                                             
         
                                             
                                
            
                          
       
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


                                                              
                           
                                                              

class DegradationEncoder(nn.Module):
                                                                          
    
                                                                            
                                           
       
    def __init__(self, in_ch=3, deg_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, deg_dim),
        )
    
    def forward(self, lq_img):
           
             
                                                  
                
                                              
           
        return self.net(lq_img)


                                                              
                                          
                                                              

class DAWPredictor(nn.Module):
                                                                             

              
                                                                                 
                                                                            
                                                        
                                                        

                                                                  
       
    def __init__(self, t_dim=64, deg_dim=32, hidden=128, num_bands=3):
        super().__init__()
        self.t_dim = t_dim
        self.num_bands = num_bands

        self.mlp = nn.Sequential(
            nn.Linear(t_dim + deg_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_bands),
            nn.Softplus(),                           
        )

                                                              
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, deg_emb):
           
             
                                      
                                                       
                
                                                                       
           
        t_emb = sinusoidal_embedding(t, self.t_dim)              
        feat = torch.cat([t_emb, deg_emb], dim=-1)                         
        weights = self.mlp(feat)                               
        return weights


                                                              
                                                       
                                                              

class TFALoss(nn.Module):
                                                                  

                                                            

          
                            
                                                       
                                                      

          
                                                                     
                                                             
       
    def __init__(self, mode='fixed', dwt_levels=2, deg_dim=32,
                 t_dim=64, hidden=128, alpha=0.5):
        super().__init__()
        self.mode = mode
        self.alpha = alpha                                                   
        self.dwt = MultiLevelDWT(levels=dwt_levels)

        if mode == 'learned':
            self.deg_encoder = DegradationEncoder(in_ch=3, deg_dim=deg_dim)
            self.weight_predictor = DAWPredictor(
                t_dim=t_dim, deg_dim=deg_dim,
                hidden=hidden, num_bands=3
            )

    def get_fixed_weights(self, t):
                                                               

               
                                                                          
                                                                             
                                                           
                                                        

                                                                         
                                                           
                                                                           
           
        B = t.shape[0]
        pi = math.pi
        w_low  = torch.cos(pi / 2 * t) + 0.3                                   
        w_mid  = torch.sin(pi * t) + 0.3
        w_high = torch.sin(pi / 2 * t) + 0.3
        return torch.stack([w_low, w_mid, w_high], dim=-1)          

    def forward(self, v_pred, v_target, t, lq_img=None):
           
             
                                                           
                                                        
                                    
                                                                       
                
                                 
                                                                        
           
                                               
        residual = v_pred - v_target                                  
        bands = self.dwt(residual)                                        

                                 
        B = v_pred.shape[0]
        err_low  = bands['low'].reshape(B, -1).pow(2).mean(dim=-1)
        err_mid  = bands['mid'].reshape(B, -1).pow(2).mean(dim=-1)
        err_high = bands['high'].reshape(B, -1).pow(2).mean(dim=-1)
        errors = torch.stack([err_low, err_mid, err_high], dim=-1)          

                        
        if self.mode == 'fixed':
            weights = self.get_fixed_weights(t)                 
        elif self.mode == 'learned':
            assert lq_img is not None, "lq_img required for learned mode"
            deg_emb = self.deg_encoder(lq_img)                        
            weights = self.weight_predictor(t, deg_emb)          
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

                         
        weighted_errors = (weights * errors).sum(dim=-1)        
        loss = weighted_errors.mean()

        return loss, weights.detach()
