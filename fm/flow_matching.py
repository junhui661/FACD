# -*- coding: utf-8 -*-
                   

import copy
from typing import Any, Iterable, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
import logging
from fm.measurements import get_noise, get_operator
from functools import partial
from fm.losses import get_flow_loss_fn
from fm.utils import calc_psnr
from fm.realsrDataPipeline import RealSRDataPipeline
from fm.freq_loss import TFALoss
import lpips

PRECISION_MAP = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}

class FM():
    def __init__(self, model=None, ema_model=None, cfg=None, device=None):
        self.cfg = cfg
        self.flow_config = cfg.fm_model
        self.config_ir = cfg.ir
        self.config_degradation = cfg.ir.degradation
        self.dataset_config = cfg.dataset
        self.network_config = cfg.network
        self.model = model
        self.ema_model = ema_model
        self.device = device
        
        self.use_ode_sampler = self.cfg.sample.use_ode_sampler
        self.psnr_batch_size = self.cfg.sample.psnr_batch_size
        self.sample_N = self.cfg.sample.sample_N
        self.ode_tol = self.cfg.sample.ode_tol
        self.eps = self.cfg.fm_model.eps
        self._T = self.cfg.fm_model.T
        self.amp_dtype = PRECISION_MAP.get(self.cfg.train.amp_dtype, torch.float16)                      
        try:
            self.flow_t_schedule = int(self.cfg.fm_model.flow_t_schedule)
        except:
            self.flow_t_schedule = self.cfg.fm_model.flow_t_schedule
                            
        self.loss_fn = get_flow_loss_fn(self.cfg.train.reduce_mean, self.cfg.train.loss_type)
        logging.info(f'sigma_y: {self.config_ir.sigma_y}')
        logging.info(f'sigma_pertubation: {self.config_ir.sigma_pertubation}')
        logging.info(f'ODE Tolerence: {self.ode_tol}')
        logging.info(f'use automatic mixed precision: {self.cfg.train.use_amp}; dtype: {self.amp_dtype}')
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.train.use_amp)
                                   
        if self.config_ir.degradation == 'realsr':
                                              
            self.real_degra_pipeline = RealSRDataPipeline(cfg=self.cfg, device=self.device)
        else:
            self.operator = get_operator(name=self.config_ir.degradation, scale_factor=self.config_ir.scale_factor, mode=self.config_ir.mode, device=self.device)
            self.noiser = get_noise(name='gaussian', sigma=self.config_ir.sigma_y)
        self.noiser_pertub = get_noise(name='gaussian_VP', sigma=self.config_ir.sigma_pertubation)
                                                           
        _need_lpips = ('lpips' in self.cfg.train.loss_type or
                       (hasattr(cfg, 'boot') and hasattr(cfg.boot, 'lambda_lpips')
                        and cfg.boot.lambda_lpips > 0))
        if _need_lpips:
            self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
            self.lpips_model.eval()
            for p in self.lpips_model.parameters():
                p.requires_grad = False
        if self.config_ir.calc_LPIPS:
            self.loss_fn_vgg = lpips.LPIPS(net='alex').to(self.device)

                                                                     
        self.use_tfa = hasattr(cfg, 'datf') and cfg.datf.get('enabled', False)
        if self.use_tfa:
            datf_cfg = cfg.datf
            self.tfa_alpha = datf_cfg.get('alpha', 0.5)
            self.tfa_loss = TFALoss(
                mode=datf_cfg.get('mode', 'fixed'),
                dwt_levels=datf_cfg.get('dwt_levels', 2),
                deg_dim=datf_cfg.get('deg_dim', 32),
                t_dim=datf_cfg.get('t_dim', 64),
                hidden=datf_cfg.get('hidden', 128),
                alpha=self.tfa_alpha,
            ).to(self.device)
            self.tfa_log_interval = datf_cfg.get('log_weights_every', 100)
            self._tfa_step_count = 0
            logging.info(f'[DATF] TFA-Loss enabled: mode={datf_cfg.get("mode", "fixed")}, '
                         f'alpha={self.tfa_alpha}, dwt_levels={datf_cfg.get("dwt_levels", 2)}')

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    def lpips_forward_wrapper(self, x, y, size=128):
                                                                                           
                                                 
        if size > 0:
                                                                                                  
            x = nn.functional.interpolate(x, size=(size, size), mode='bilinear', antialias=True)
            y = nn.functional.interpolate(y, size=(size, size), mode='bilinear', antialias=True)
        return self.lpips_model(x, y)

    def model_forward_wrapper(
        self,
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> Tensor:
                                        
        kwargs = {} if kwargs is None else kwargs
        label = kwargs['label'] if 'label' in kwargs else None
        augment_labels = kwargs['augment_labels'] if 'augment_labels' in kwargs else None
        x = torch.cat([x, self.cond], dim=1) if self.flow_config.use_cond else x
        model_output = model(x, t*999, label, augment_labels)
        model_output = model_output[0] if isinstance(model_output, tuple) else model_output
        return model_output[:, :3]

    def get_train_tuple_IR(self,
                            batch,
                            mask: Tensor = None,
                            **kwargs: Any,):
                                                    
                                                                         
        x = batch['gt'].to(self.device)
        yn_ = batch['lq'].to(self.device)
        self.cond = yn_.detach().clone()
        yn = self.noiser_pertub(yn_)                        
        self.x1 = x
        self.x0 = yn

    def get_interpolations(self,
                        data: Tensor,
                        noise: Tensor,
                        **kwargs: Any,):
                                                        
                          
        if self.flow_t_schedule == 't0':                            
            self.t = torch.zeros((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 't1':                                             
            self.t = torch.ones((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 't0t1':                                         
            self.t = torch.randint(0, 2, (data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif self.flow_t_schedule == 'uniform':                                         
            self.t = torch.rand((data.shape[0],), device=data.device) * (self.T - self.eps) + self.eps
        elif type(self.flow_t_schedule) == float:                                          
            self.t = torch.ones((data.shape[0],), device=data.device) * self.flow_t_schedule
        elif type(self.flow_t_schedule) == int:           
            self.t = torch.randint(0, self.flow_t_schedule, (data.shape[0],), device=data.device) * (self.T - self.eps) / self.flow_t_schedule + self.eps
        else:
            assert False, f'flow_t_schedule {self.flow_t_schedule} Not implemented'
                                                            
        self.xt = torch.einsum('b,bijk->bijk', self.t, data) + torch.einsum('b,bijk->bijk', (1 - self.t), noise)                             

    def pred_tuple(self, **kwargs: Any,) -> Tuple[Tensor, Tensor]:
                                     
        pred = self.model_forward_wrapper(
            self.model,
            self.xt,
            self.t,
            **kwargs,
        )
                                       
        target = self.x1 - self.x0
        return pred, target
    
    def train_step(self, batch, augment_pipe=None, mode='train', **kwargs: Any,):
                                      
                    
        '''
        batch: Clean data.
        '''
                                                
        self.get_train_tuple_IR(batch, mode=mode)
                                   
        self.get_interpolations(self.x1, self.x0)
                                    
        with torch.autocast(device_type="cuda", enabled=self.cfg.train.use_amp, dtype=self.amp_dtype):
                                                       
            predicted, target = self.pred_tuple(**kwargs)
                                                
            loss_spatial = self.loss_fn(self, predicted, target)

                                                        
            if self.use_tfa:
                lq_img = self.cond if hasattr(self, 'cond') else None
                loss_tfa, tfa_weights = self.tfa_loss(
                    predicted, target, self.t, lq_img=lq_img
                )
                                                          
                loss = (1 - self.tfa_alpha) * loss_spatial + self.tfa_alpha * loss_tfa

                         
                self._tfa_step_count += 1
                if self._tfa_step_count % self.tfa_log_interval == 0:
                    w_mean = tfa_weights.mean(dim=0)        
                    logging.info(
                        f'[DATF] step={self._tfa_step_count} | '
                        f'L_spatial={loss_spatial.item():.6f} | '
                        f'L_TFA={loss_tfa.item():.6f} | '
                        f'w_low={w_mean[0]:.3f} w_mid={w_mean[1]:.3f} w_high={w_mean[2]:.3f}'
                    )
            else:
                loss = loss_spatial
        return loss

                                                                                        
    @torch.no_grad()
    def test_split_fn(self, x_0, cond, refield=32, min_size=256, modulo=1, sample_fn=None, t=0.001):
           
              
                               
                                                     
                                                                       
                                                              
                          
           
        h, w = x_0.size()[-2:]
        if h*w <= min_size**2:
            x_0 = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(x_0)
            cond = torch.nn.ReplicationPad2d((0, int(np.ceil(w/modulo)*modulo-w), 0, int(np.ceil(h/modulo)*modulo-h)))(cond)
            E, nfe = self.image_restoration_wrapper(x_0, cond, sample_fn, t)
        else:
            top = slice(0, (h//2//refield+1)*refield)
            bottom = slice(h - (h//2//refield+1)*refield, h)
            left = slice(0, (w//2//refield+1)*refield)
            right = slice(w - (w//2//refield+1)*refield, w)
            x_0s = [x_0[..., top, left], x_0[..., top, right], x_0[..., bottom, left], x_0[..., bottom, right]]
            conds = [cond[..., top, left], cond[..., top, right], cond[..., bottom, left], cond[..., bottom, right]]

            if h * w <= 4*(min_size**2):
                Es = []
                nfe = 0
                for i in range(4):
                    E, n = self.image_restoration_wrapper(x_0s[i], conds[i], sample_fn, t)
                    Es.append(E)
                    nfe += n
                nfe = nfe / 4
            else:
                                                                                                                                                   
                Es = []
                nfe = 0
                for i in range(4):
                    E, n = self.test_split_fn(x_0s[i], conds[i], refield=refield, min_size=min_size, modulo=modulo, sample_fn=sample_fn, t=t)
                    Es.append(E)
                    nfe += n
                nfe = nfe / 4

            b, c = Es[0].size()[:2]
            E = torch.zeros(b, c, h, w).type_as(x_0)

            E[..., :h//2, :w//2] = Es[0][..., :h//2, :w//2]
            E[..., :h//2, w//2:w] = Es[1][..., :h//2, (-w + w//2):]
            E[..., h//2:h, :w//2] = Es[2][..., (-h + h//2):, :w//2]
            E[..., h//2:h, w//2:w] = Es[3][..., (-h + h//2):, (-w + w//2):]
        return E, nfe


    @torch.no_grad()
    def image_restoration_(self, x_0, yn, sample_fn):
        samples = []
        nfe = []
        for i in range(int(np.ceil(x_0.shape[0] / self.psnr_batch_size))):
            batch_x0 = x_0[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
            self.cond = yn.detach()[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
            sample, n = sample_fn(self.model, z=batch_x0)
            samples.append(sample.cpu())
            nfe.append(n)
        samples = torch.cat(samples, dim=0)
        return samples, np.mean(nfe)

    @torch.no_grad()
    def image_restoration_t_(self, x_0, yn, t):
        samples = []
        for i in range(int(np.ceil(x_0.shape[0] / self.psnr_batch_size))):
            batch_x0 = x_0[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]                      
            vec_t = torch.ones(batch_x0.shape[0],).to(self.device) * t
            self.cond = yn.detach()[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size]
                                                           
            with torch.no_grad():
                v_pred = self.model_forward_wrapper(self.model, batch_x0, vec_t)
                sample = batch_x0 + (self.T - self.eps) * v_pred                                        
            samples.append(sample.cpu())
        samples = torch.cat(samples, dim=0)
        return samples
    
    @torch.no_grad()
    def image_restoration_wrapper(self, x_0, yn, sample_fn=None, t=0.001):
        if self.cfg.sample.__has_attr__('use_one_step') and self.cfg.sample.use_one_step:
            samples = self.image_restoration_t_(x_0, yn, t=t)
            nfe = 1
        else:
            samples, nfe = self.image_restoration_(x_0, yn, sample_fn)
        return samples, nfe

    @torch.no_grad()
    def image_restoration(self, val_batch, sample_fn=None, t=0.001):
                                                             
        data = val_batch
        yn = data['lq'].to(self.device)
        y_LR = data['lq_true'].to(self.device)
        x_0 = self.noiser_pertub(yn)                        
        LR = (y_LR, yn, x_0)
                                                                              
        samples, nfe = self.test_split_fn(x_0, yn, sample_fn=sample_fn, t=t)
        psnr = calc_psnr(val_batch['gt'].cpu(), samples.cpu())
        if self.config_ir.calc_LPIPS:
            with torch.no_grad():
                lpips_scores = 0
                for i in range(int(np.ceil(val_batch['gt'].shape[0] / self.psnr_batch_size))):
                    batch_samples = samples[i*self.psnr_batch_size:(i+1)*self.psnr_batch_size].to(self.device)
                    batch_val_batch = val_batch['gt'][i*self.psnr_batch_size:(i+1)*self.psnr_batch_size].to(self.device)
                    lpips_score = self.loss_fn_vgg(batch_samples, batch_val_batch)
                    lpips_score = lpips_score.cpu().detach().numpy().mean()           
                    lpips_scores += (lpips_score * batch_samples.shape[0])
                lpips_score = lpips_scores / val_batch['gt'].shape[0]
        else:
            lpips_score = 0
        return psnr, lpips_score, samples, LR, np.mean(nfe)
