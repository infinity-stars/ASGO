import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from LLMmodels.utils.data_utils import load_train_test_loaders
from LLMmodels.utils.utils import *
from optimizers.utils import *
from optimizers.utils.utils import get_optimizer_and_lr_scheduler
import wandb
import datetime
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record



@record
@hydra.main(config_path="config", config_name="train_config", version_base="1.2")
def main(cfg: DictConfig):
    # Start DDP if required in training config
    if cfg.train.DDP:
        # Get local_rank from environment variable set by torchrun
        # Initialize the process group
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=global_rank
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(cfg.train.device)

    # Wandb log in and initialization
    if rank == 0:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        run = wandb.init(
            entity = cfg.train.wandb_entity,
            project=cfg.train.wandb_project,
            name=cfg.train.wandb_name + f"_{timestamp}",
            reinit=True,
            resume="never",  # 显式禁止恢复
            dir=hydra_cfg.runtime.output_dir,
        )
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.config.update(config_dict)
    else:
        run = None

    seed_everything(cfg.train.seed + rank)
    # Load the model and DDP model
    model = get_LLM_network(cfg.model).to(device)
    if rank == 0:
        print(f"[Rank {local_rank}] Model created. Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6} M")
    
    if cfg.train.DDP:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # Load the dataset and loss function
    train_loader, val_loader = load_train_test_loaders(cfg)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Don't need to do special initialization for Optimizer
    optimizer_class, lr_scheduler_list = get_optimizer_and_lr_scheduler(
        model = model, cfg = cfg, total_steps=cfg.train.train_steps
    )


    if rank == 0:
        print("Training process start!")
    for step in range(cfg.train.train_steps):
        model.train()
        loss_acc = 0
        for micro_step in range(cfg.train.gradient_accumulation_steps):
            inputs, targets = next(train_iter)
            inputs, targets = inputs.to(device), targets.to(device)
            if cfg.train.DDP:
                model.require_backward_grad_sync = (micro_step == cfg.train.gradient_accumulation_steps - 1)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1,outputs.size(-1)), targets.view(-1))
            loss /= cfg.train.gradient_accumulation_steps
            loss_acc += loss.detach()
            loss.backward()
        del inputs, targets, outputs, loss
        if cfg.train.global_clip_value != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.global_clip_value)
        optimizer_class.step()

        optimizer_class.zero_grad(set_to_none=True)
        for lr_scheduler in lr_scheduler_list:
            lr_scheduler.step()
        
        
        if cfg.train.DDP:
            reduced_loss = loss_acc.clone()
            dist.reduce(reduced_loss, dst = 0)
            if rank == 0:
                loss_item = reduced_loss.item() / world_size
        else:
            loss_item = loss_acc.item()
        # if step < 5:
        #     verify_model_sync(model, rank, world_size)
        if rank == 0:
            if cfg.train.verbose and (step + 1) % (cfg.train.train_verbose_step) == 0:
                print(f'\r[{cfg.optimizer.optimizer_name}] Step: {step + 1}, '
                    f'step_loss: {loss_item:.3f}, ', 
                    end='\r\n', flush=True)
            lr_list = []
            for optimizer in optimizer_class.optimizers:
                lr_list.append(optimizer.param_groups[0]['lr'])
            run.log({"Train Step Loss": loss_item, "embedding_lr": lr_list[0], "other_lr": lr_list[1]}, step=step)
        if cfg.train.DDP:
            dist.barrier()
        # Do the validation on single GPU
        if ((step + 1) % cfg.train.val_iters == 0) and (rank == 0):
            val_loss = 0
            if cfg.train.DDP:
                val_model = model.module
            else:
                val_model = model
            model.eval()
            print('Validating...')
            with torch.no_grad():
                for val_step in range(cfg.train.val_steps):
                    inputs, targets = next(val_iter)
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = val_model(inputs)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    val_loss += loss.item()
                    if (val_step + 1) % 16 == 0:
                        print(f'Validating: \r[{cfg.optimizer.optimizer_name}] Step: {val_step}, '
                              f'val step loss: {loss.item():.3f}, '
                              f'val avg loss: {val_loss/(val_step + 1):.3f}',
                              end='\r\n', flush=True)
                run.log({"Test Loss": val_loss / cfg.train.val_steps}, step=step)
        
        # Wait for the validation to finish
        if cfg.train.DDP:
            dist.barrier()


if __name__ == "__main__":
    main()