# -*- coding: utf-8 -*-
                   

import os
import sys
join = os.path.join
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from fm.utils import (parse_args_and_config,
                      seed_everywhere,
                      restore_checkpoint)
from fm.image_datasets import ImageDataset
from fm import sampling as sampling
from models import create_model
from models.ema import ExponentialMovingAverage
from fm import FM
from datetime import datetime
import logging
from matplotlib import pyplot as plt

def save_batch_LR_SR(val_batch, samples, img_name, vis_num):
    sample_RLH = torch.cat([val_batch[:vis_num].cpu(), samples[:vis_num].cpu()], dim=3)
    sample_RLH = (sample_RLH + 1) / 2
    save_image(sample_RLH, img_name, nrow=1)

def main():
                                                                          
    config = parse_args_and_config()
                           
    dataset_config = config.dataset
    flow_config = config.fm_model
    network_config = config.network
    sample_config = config.sample
    save_eval_images = (not sample_config.__has_attr__('save_images')) or sample_config.save_images
    dataset_config.in_channels = network_config.in_channels
    network_config.img_size = dataset_config.img_size
    network_config.num_classes = dataset_config.num_classes
                                                                                                                         
    network_config.world_size = torch.cuda.device_count()
    network_config.use_cond = flow_config.use_cond
    
                                                                             
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    if 'checkpoints/' in sample_config.pre_train_model:
        work_dir = sample_config.pre_train_model.split('checkpoints')[0]
    else:
        work_dir = 'results'
        os.makedirs(work_dir, exist_ok=True)
    model_path = join(work_dir, f'eval_samples/{run_id}')
    os.makedirs(model_path, exist_ok=True)
                                
    seed_everywhere(config.seed)

                  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

                                                                          
    gfile_stream = open(f'{work_dir}/eval.log', 'a+')
    handler = logging.StreamHandler(gfile_stream)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(filename)s - %(asctime)s - %(levelname)s --> %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    logger.setLevel('INFO')

                                                                          
    unet = create_model(network_config)
    unet.change_input_conv()
    unet = unet.to(device)
    ema = ExponentialMovingAverage(unet.parameters(), decay=network_config.ema_rate)
    state = dict(optimizer=None, model=unet, ema=ema, step=0)
    
    assert sample_config.pre_train_model
    state = restore_checkpoint(sample_config.pre_train_model, state, device, ema_only=True)
    logger.info(f'loading model from {sample_config.pre_train_model}')
    
                                                                           
    flow = FM(model=unet, ema_model=ema, cfg=config, device=device)
    sampling_fn = sampling.get_flow_sampler(flow, device=device, use_ode_sampler=sample_config.use_ode_sampler)

                                                                           
    ema.store(unet.parameters())
    unet.eval()
    ema.copy_to(unet.parameters())
    
                                                                           
    if (not config.fm_model.use_cond) and config.ir.sigma_pertubation == 1.:
                            
        batch_shape = (sample_config.psnr_batch_size, network_config.in_channels, dataset_config.img_size, dataset_config.img_size)
        flow.cond = torch.zeros(batch_shape).to(device)
        for i in tqdm(range(sample_config.num_sample // sample_config.psnr_batch_size + 1)):
            x0 = torch.randn(batch_shape).to(device)
            with torch.no_grad():
                sample, n = sampling_fn(unet, z=x0)
            for img_idx in range(sample.shape[0]):
                if i*sample_config.psnr_batch_size + img_idx >= sample_config.num_sample:
                    break
                img_name = join(model_path, f"sample_{i*sample_config.psnr_batch_size+img_idx}_seed{config.seed}.{sample_config.file_ext}")
                save_image(sample[img_idx:img_idx+1]/2+0.5, img_name, nrow=1)
            logger.info(f"sample batch {i} --> nfe: {n}")
                                                                        
                                                                                                          
                                                                                               
    else:
                             
                        
        if save_eval_images:
            os.makedirs(model_path + 'LR', exist_ok=True)
            os.makedirs(model_path + 'LR_yn', exist_ok=True)
            os.makedirs(model_path + 'LR_input_perturb', exist_ok=True)
            os.makedirs(model_path + 'HR', exist_ok=True)
                            
        img_dataset = ImageDataset(dataset_config, phase='val')
        data_loader = torch.utils.data.DataLoader(img_dataset,
                                                    batch_size=sample_config.psnr_batch_size,
                                                    shuffle=False,
                                                    num_workers=dataset_config.num_workers,
                                                    pin_memory= True)
        logger.info(f'evaluate from: {dataset_config.val_path}; length of img_dataset: {len(img_dataset)}')
        logger.info(
            f'eval protocol: center_crop_val={dataset_config.center_crop_val}; '
            f'random_crop_val={dataset_config.random_crop_val}; '
            f'use_one_step={getattr(sample_config, "use_one_step", False)}; '
            f'one_step_t={getattr(sample_config, "one_step_t", None)}; '
            f'psnr_batch_size={sample_config.psnr_batch_size}; '
            f'save_images={save_eval_images}'
        )
        if len(img_dataset) == 0:
            raise RuntimeError(
                f"No validation images found under dataset.val_path={dataset_config.val_path}. "
                "Check that the path exists and contains png/jpg/jpeg/bmp/tif/tiff files."
            )
        all_psnr = 0
        all_lpips = 0
        num_samples = 0
        nfe = 0
        for i, (val_batch, label_dic) in tqdm(enumerate(data_loader), total=len(data_loader)):
                                                         
            val_batch_dict = {}
            y = flow.operator.forward(val_batch, mask=None)
            yn = flow.noiser(y)
            val_batch_dict['lq_true'] = yn.clone()
            if config.ir.scale_factor > 1:
                yn_ = flow.operator.transpose(yn, mask=None)
            val_batch_dict['gt'] = val_batch
            val_batch_dict['lq'] = yn_.detach().clone()

            num_samples += val_batch.shape[0]
                                                                                                  
            t_in = sample_config.one_step_t if sample_config.__has_attr__('use_one_step') and sample_config.use_one_step else 0.001
            psnr, lpips_score, samples, LR, n = flow.image_restoration(val_batch_dict, sampling_fn, t_in)
            y_LR, yn, x_0 = LR
            all_psnr += psnr * val_batch.shape[0]
            all_lpips += lpips_score * val_batch.shape[0]
            nfe += n * val_batch.shape[0]
            if save_eval_images:
                for img_idx in range(val_batch.shape[0]):
                    img_name = join(model_path, f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}_seed{config.seed}.{sample_config.file_ext}")
                    save_image(samples[img_idx:img_idx+1]/2+0.5, img_name, nrow=1)
                    img_name_LR = join(model_path + 'LR', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                    save_image(y_LR[img_idx]/2+0.5, img_name_LR.replace(f'_{flow.use_ode_sampler}','LR'), nrow=1)
                    img_name_LR_yn = join(model_path + 'LR_yn', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                    save_image(yn[img_idx]/2+0.5, img_name_LR_yn.replace(f'_{flow.use_ode_sampler}','LR_yn'), nrow=1)
                    img_name_LR_p = join(model_path + 'LR_input_perturb', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                    save_image(x_0[img_idx]/2+0.5, img_name_LR_p.replace(f'_{flow.use_ode_sampler}','LR_perturb'), nrow=1)
                    img_name_Hr = join(model_path + 'HR', f"{label_dic['img_name'][img_idx]}_{flow.use_ode_sampler}.{sample_config.file_ext}")
                    save_image(val_batch[img_idx:img_idx+1]/2+0.5, img_name_Hr.replace(f'_{flow.use_ode_sampler}','HR'), nrow=1)
            logger.info(f"batch {i} --> psnr: {psnr}; lpips: {lpips_score}; nfe: {n}; ave PSNR {all_psnr / num_samples}; ave lpips {all_lpips / num_samples}")
        all_psnr /= num_samples
        all_lpips /= num_samples
        nfe /= num_samples
        logger.info(f"[EVAL] --> steps: {state['step']}; psnr: {all_psnr}; lpips: {all_lpips}; nfe: {nfe}")
    
                   
                                   
                                                                
                                                 
    if save_eval_images:
        logger.info(f"evaluation done! saved to {model_path}\n")
    else:
        logger.info(f"evaluation done! images not saved; log dir {model_path}\n")

                               
             
                       
                 
                                                                                           
                        
                                                                                            
                                         
                                                 
                                                           
                                                              
                         
                                         
                                              
                                  
                    
                             
                                                                                                
                                              
                       
                        
                                           
                                                                                          
                                                             
                                                                      
                                                   
                                              
                                             
                                                        
                                                                                                                

                                
                                  
                    
              
                       
                                                                                           

                                                                                            
               
                                                              
                         
                                         
                                              
                                  
                    
                                 
                           
                                                                                                     
                
                                         
                                                            
                                                            
                                            
                                                                            

                                 
                                                     
                                                  
                                                                                             

                                     
                        
                               
                              
                    
                                                                                                                                                                                   
                
                                
                 
    
if __name__ == "__main__":
    sys.exit(main())
