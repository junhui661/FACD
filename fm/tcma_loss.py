# -*- coding: utf-8 -*-
   
                                                               
                                          

                                              

                                                  
                                                 
                                           
                                                         
                                                                    
                                                                     
                                                                                
                                                                                       

                                                                 
                                                           
   
   

from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_consistency_loss(
    student_forward,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    T_val: float = 1.0,
    eps_val: float = 0.001,
    tau_offset: float = 0.1,
    t_min: float = 0.01,
    t_max: float = 0.99,
) -> torch.Tensor:
       
                                                

                                                                        
                                                                      

                                                                             
                                                                                 
                                                                                         

                                                                           
                                                             

                                                                         

         
                                                                           
                                      
                                                                             
                                    
                           
                                 
                                                            
                                 

            
                                                                           
       
    scale = T_val - eps_val          

                                   
    with torch.no_grad():
        v_t = student_forward(x_0, t).detach()

                               
    tau = (t - tau_offset).clamp(t_min, t_max)
    v_tau = student_forward(x_0, tau)

                                                                                  
    loss = F.mse_loss(v_tau * scale, v_t * scale)
    return loss


def marginal_velocity_alignment_loss(
    student_forward,
    aux_forward,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sample_t_fn,
    T_val: float = 1.0,
    eps_val: float = 0.001,
    teacher_forward=None,
) -> tuple[torch.Tensor, torch.Tensor]:
       
                                                

                                                                   
                                                                          
                                                                          

                                 
                                                            
                                                       
                                                           
                                                                      

                                                                 
                                                                    
                                                                     
                 

                                                                        
                                                                        
                                                               
                                                                     
                            

         
                                                                           
                                                                           
                                                        
                                                     
                                                         
                                              
                                      

            
                                                  
                                                     
       
    B = x_0.shape[0]
    device = x_0.device
    scale = T_val - eps_val          

                                                     
    with torch.no_grad():
        v_t = student_forward(x_0, t)
        x1_fake = (x_0 + scale * v_t).clamp(-1.0, 1.0)

                                                                 
    t_prime = sample_t_fn(B, device)

                                                                      
                                                                  
                                                                    
                                                          
                                                                   
                                                                
                                                                    
                                                           
    v_aux_target = (x1_fake.detach() - x_0) / scale

    v_aux_pred = aux_forward(x_0, t_prime)
    aux_loss = F.mse_loss(v_aux_pred, v_aux_target.detach())

                                                         
                                                                 
                                                                               
                                                                         
    scale = T_val - eps_val
    with torch.no_grad():
        if teacher_forward is not None:
            u_real = teacher_forward(x_0, t_prime).detach()
        else:
                                                                
            u_real = ((x_1 - x_0) / scale).detach()
        u_fake = aux_forward(x_0, t_prime).detach()

                                       
    v_main = student_forward(x_0, t_prime)

                                                          
                                                                             
                                                                               
    correction = u_real - u_fake
    align_target = (u_real + 0.5 * correction).detach()

    alignment_loss = F.mse_loss(v_main, align_target)

    return aux_loss, alignment_loss
