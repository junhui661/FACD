import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path

import albumentations

import torch.nn.functional as F
from torch.utils import data as data

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x[:-1] for x in ff.readlines()])

    return out

@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
                                          
                                                                                     

                                                         
                                                                                      
                                                                                            

         
                                                                              
                                                     
                                                            
                                                               
                                                   
                                                                                                        
                                                 
       

    def __init__(self, opt, mode='training'):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

                                       
        self.paths = []
        if 'dir_paths' in opt:
            for current_dir in opt['dir_paths']:
                for current_ext in opt['im_exts']:
                    self.paths.extend(sorted([str(x) for x in Path(current_dir).glob(f'**/*.{current_ext}')]))
        if 'txt_file_path' in opt:
            for current_txt in opt['txt_file_path']:
                self.paths.extend(readline_txt(current_txt))
        if 'length' in opt:
            self.paths = random.sample(self.paths, opt['length'])

                                                 
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']                                      
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']                                                   
        self.betap_range = opt['betap_range']                                      
        self.sinc_prob = opt['sinc_prob']                                    

                                                  
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

                             
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]                                   
        self.kernel_range2 = [x for x in range(3, opt['blur_kernel_size2'], 2)]                                   
                                                                               
                                                              
        self.pulse_tensor = torch.zeros(opt['blur_kernel_size2'], opt['blur_kernel_size2']).float()
        self.pulse_tensor[opt['blur_kernel_size2']//2, opt['blur_kernel_size2']//2] = 1

        self.mode = mode
        self.rescale_gt = opt['rescale_gt']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

                                                                                            
                                                                             
        gt_path = self.paths[index]
                                                              
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                                                             
            except:
                                            
                                                                                            
                                             
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)                                             
                   
                       
            finally:
                retry -= 1
        if self.mode == 'testing':
            if not hasattr(self, 'test_aug'):
                self.test_aug = albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=self.opt['gt_size']),
                    albumentations.CenterCrop(self.opt['gt_size'], self.opt['gt_size']),
                    ])
            img_gt = self.test_aug(image=img_gt)['image']
        elif self.mode == 'training':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

        if self.mode == 'training':
                                                                                                      
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

                                
                                                                    
            h, w = img_gt.shape[0:2]
            if self.rescale_gt:
                crop_pad_size = max(min(h, w), self.opt['gt_size'])
            else:
                crop_pad_size = self.opt['crop_pad_size']
                 
                                                        
                                                   
                                                   
                                                                                                 
            while h < crop_pad_size or w < crop_pad_size:
                pad_h = min(max(0, crop_pad_size - h), h)
                pad_w = min(max(0, crop_pad_size - w), w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                h, w = img_gt.shape[0:2]
                  
            if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
                h, w = img_gt.shape[0:2]
                                                          
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)
                img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            if self.rescale_gt and crop_pad_size != self.opt['gt_size']:
                img_gt = cv2.resize(img_gt, dsize=(self.opt['gt_size'],)*2, interpolation=cv2.INTER_AREA)
        elif self.mode == 'testing':
            pass
        else:
            raise ValueError(f'Unexpected value {self.mode} for mode parameter')

                                                                                                              
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opt['sinc_prob']:
                                                                          
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
                    
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

                                                                                                               
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

                    
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

                                                                                                             
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

                                                 
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        out_dict = {'img_name': os.path.splitext(os.path.basename(gt_path))[0]}
        return return_d, out_dict

    def __len__(self):
        return len(self.paths)

    def degrade_fun(self, conf_degradation, im_gt, kernel1, kernel2, sinc_kernel):
        if not hasattr(self, 'jpeger'):
            self.jpeger = DiffJPEG(differentiable=False)                                       

        ori_h, ori_w = im_gt.size()[2:4]
        sf = conf_degradation.sf

                                                                                         
              
        out = filter2D(im_gt, kernel1)
                       
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                conf_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, conf_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(conf_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
                   
        gray_noise_prob = conf_degradation['gray_noise_prob']
        if random.random() < conf_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=conf_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=conf_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
                          
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)                                                                         
        out = self.jpeger(out, quality=jpeg_p)

                                                                                          
              
        if random.random() < conf_degradation['second_order_prob']:
            if random.random() < conf_degradation['second_blur_prob']:
                out = filter2D(out, kernel2)
                           
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    conf_degradation['resize_prob2'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, conf_degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(conf_degradation['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                    mode=mode,
                    )
                       
            gray_noise_prob = conf_degradation['gray_noise_prob2']
            if random.random() < conf_degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=conf_degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=conf_degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                    )

                                                  
                                                                                                       
                           
                                 
                                                             
                                                             
                                                                                                      
        if random.random() < 0.5:
                                                 
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf, ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
                              
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
                              
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
                                                 
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf, ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

                         
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq':im_lq.contiguous(), 'gt':im_gt}
