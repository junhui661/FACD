# -*- coding: utf-8 -*-
   
                                                                        

                        
                                                             
  
                                                         
                                                                           
                                                                        
  
                                                      
                                                    
                                                                    

                                      
                                                                 
                                                             
                                                                            

      
                                                                  
                                                   
   

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianSplit(nn.Module):
       
                                                                 

                                          
                                                                       
                                                                       
                                                                            
                                                                        
                                                       

                                                                    
                                                                            

                                                                           
                                                    
                                                    
       
    def __init__(self):
        super().__init__()
                                                                      
        k1d = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.]) / 256.
        k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)           
        self.register_buffer('kernel', k2d.unsqueeze(0).unsqueeze(0))             

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C, 1, H, W)
                                                                             
        low_flat  = F.conv2d(x_flat, self.kernel, padding=4, stride=1)
        high_flat = x_flat - low_flat
        low  = low_flat.reshape(B, C, H, W)
        high = high_flat.reshape(B, C, H, W)
                                                                          
        zero = torch.zeros_like(high[:, :, :1, :1])                          
        return low, high, zero, zero


def get_freq_weights(t, mode='cosine'):
       
                                                     

                            
                                                     
                                                                             
                                                                              

                                                                  
                                                         
                                                          
                                                         

         
                                  
                                                          
            
                                           
       
    if mode == 'cosine':
                                                                       
                                                              
                                                              
        pi = math.pi
        w_low = 0.5 + 0.3 * torch.cos(pi * t)                    
        w_high = 1.5 + 0.5 * torch.sin(pi / 2 * t)               
    elif mode == 'linear':
        w_low = 1.0 - 0.5 * t                
        w_high = 1.0 + 1.0 * t               
    elif mode == 'constant':
                                                             
        w_low = torch.ones_like(t) * 0.5
        w_high = torch.ones_like(t) * 2.0
    else:
        w_low = torch.ones_like(t)
        w_high = torch.ones_like(t)

    return w_low.view(-1, 1, 1, 1), w_high.view(-1, 1, 1, 1)


class LearnableScheduleMLP(nn.Module):
       
                                                                    

                                                                     
                                                     
                                                                
                                                                
                                                                      
                                                                 
                                                                        
                                                            
                                                           

                                                                        
                                                                  
       
    def __init__(self, hidden=32, num_freq=4,
                 w_low_min=0.1, w_low_max=1.0,
                 w_high_min=0.5, w_high_max=2.5):
        super().__init__()
        self.num_freq = num_freq
                                                               
        self.register_buffer('w_low_min',  torch.tensor(float(w_low_min)))
        self.register_buffer('w_low_max',  torch.tensor(float(w_low_max)))
        self.register_buffer('w_high_min', torch.tensor(float(w_high_min)))
        self.register_buffer('w_high_max', torch.tensor(float(w_high_max)))

        in_dim = 2 * num_freq                                                 
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
                                                                     
                                                    
                                                                               
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def _embed(self, t):
                                      
        freqs = 2.0 ** torch.arange(self.num_freq, device=t.device, dtype=t.dtype)
        ang = t.unsqueeze(-1) * freqs * math.pi           
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, t):
        emb = self._embed(t)                           
        out = self.net(emb)                           
        sig = torch.sigmoid(out)                                
        w_low  = self.w_low_min  + (self.w_low_max  - self.w_low_min)  * sig[..., 0]
        w_high = self.w_high_min + (self.w_high_max - self.w_high_min) * sig[..., 1]
        return w_low.view(-1, 1, 1, 1), w_high.view(-1, 1, 1, 1)


class FreqAwareDistillLoss(nn.Module):
       
                                                  
    
                                                                      
                                                
                                                    
    
                                         
                                 
                                                            
    
                                                                     
                                                          
                                                                   
                                                          
                                                                      
       
    def __init__(self, weight_mode='cosine', alpha=0.3):
        super().__init__()
        self.dwt = GaussianSplit()                                                   
        self.weight_mode = weight_mode
        self.alpha = alpha                                          
                                                                              
                                                                             
                                                                   
        if weight_mode == 'learnable':
            self.learnable_schedule = LearnableScheduleMLP()
        else:
            self.learnable_schedule = None

    def freq_mse(self, pred, target, t):
           
                               

             
                                                       
                                                      
                            
                
                        
                                                       
           
        residual = pred - target
                                                                        
        ll, lh, hl, hh = self.dwt(residual)

                      
                                                                               
                                                                   
        loss_low  = ll.pow(2).mean(dim=[1, 2, 3])         
        loss_high = lh.pow(2).mean(dim=[1, 2, 3])                         

                                
        if self.learnable_schedule is not None:
            w_low, w_high = self.learnable_schedule(t)
        else:
            w_low, w_high = get_freq_weights(t, self.weight_mode)
        w_low = w_low.squeeze()          
        w_high = w_high.squeeze()        
        
                      
        loss = (w_low * loss_low + w_high * loss_high).mean()
        
        info = {
            'loss_low': loss_low.mean().item(),
            'loss_high': loss_high.mean().item(),
            'w_low': w_low.mean().item(),
            'w_high': w_high.mean().item(),
        }
        return loss, info
    
    def forward(self, pred, target, t=None, reduction='mean'):
           
                                                        
        
                                                      
           
                      
        if reduction == 'mean':
            loss_std = F.mse_loss(pred, target, reduction='mean')
        else:
            loss_std = F.mse_loss(pred, target, reduction='none')
            loss_std = loss_std.view(pred.shape[0], -1).mean(dim=1)
        
        if t is None or self.alpha == 0:
            return loss_std, {'loss_std': loss_std.item() if isinstance(loss_std, torch.Tensor) else loss_std}
        
                           
        loss_freq, info = self.freq_mse(pred, target, t)
        info['loss_std'] = loss_std.item()
        info['loss_freq'] = loss_freq.item()
        
               
        if reduction == 'mean':
            loss = (1 - self.alpha) * loss_std + self.alpha * loss_freq
        else:
            loss = (1 - self.alpha) * loss_std.mean() + self.alpha * loss_freq
        
        return loss, info
