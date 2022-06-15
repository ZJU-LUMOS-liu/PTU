import argparse
import ruamel.yaml as yaml
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import datetime
import random
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset import create_dataset,create_loader
from model.Bartbased_coin_finetune import text_baseline
from optim import create_optimizer
from scheduler import create_scheduler
import json

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('text_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(b_Prompt,b_Predict) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 

        text_loss = model(b_Prompt,b_Predict,device)                  
        loss = text_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(text_loss=text_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 

def eval(model, data_loader, device, config):
    model.eval()    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    start_time = time.time()
    #begin evaluation

    for i,item in enumerate (data_loader):
        #print(item)
        Test=item[0]
        Inputs=model.pretrained_tokenizer(Test, truncation=True, padding=True,return_tensors='pt')
        
        #beam_search used in generation
        summary_ids = model.pretrained_model.generate(Inputs['input_ids'].to(device,non_blocking=True), num_beams=4, max_length=30, early_stopping=True)
        print("Current Prompts are:")
        print(item[0])
        print("/n")
        print("Predicted labels are:")
        print([model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
        print("/n")
        print("Real predictions are:")
        print(item[1])
        print("/n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    
    Invariance=0
    return Invariance


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating coin_text dataset")
    train_dataset, test_dataset = create_dataset('coin_text', config)  
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        #samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader([train_dataset, test_dataset],samplers,
                                                batch_size=[config['batch_size_train']]+[config['batch_size_test']],
                                                num_workers=[4,4],
                                                is_trains=[True, False], 
                                                collate_fns=[None,None])   
    
    #for i,item in enumerate (test_loader):
    #    print(item)


    #### Model #### 
    print("Creating model")
    model = text_baseline(config=config, pretrianed_model=args.pretrained_model)
    
    '''never need a checkpoint in this model'''
    if args.checkpoint:    
        '''

        load checkpoint if given

        '''

        pass
           
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Start training")
    start_time = time.time()   
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)  
            
        if(epoch % 5 == 0):
            evaluation_score = eval(model_without_ddp, test_loader, device, config)
    
        '''
        if utils.is_main_process():  
      
            #val_result = itm_eval(score_val, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             #**{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, f"epoch_{epoch}_log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             #**{f'test_{k}': v for k, v in test_result.items()},                  
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))  

        '''       
        #if args.evaluate: 
        #    break

        #lr_scheduler.step(epoch+warmup_steps+1)  
        #dist.barrier()     
        #torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/coin_text.yaml')
    parser.add_argument('--pretrained_model', default='BartForConditionalGerneration')
    parser.add_argument('--output_dir', default='./output/text_baseline')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    #copy config file into the output directory.
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)