import torch.optim as optim
import torch

class WarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, init_lr, ratio = 0.01, last_epoch = -1):
        """
        Args: 
            Optimizer: PyTorch optimizer object
            warmup_steps: Number of warmup steps
            init_ratio: Initial ratio of the learning rate
            last_step: Last epoch
        """

        self.warmup_steps = warmup_steps
        self.init_lr = init_lr * ratio
        self.lr = init_lr
        super().__init__(optimizer, last_epoch = last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            lr = self.init_lr + (self.lr - self.init_lr) * (self.last_epoch / self.warmup_steps)
        else:
            lr = self.lr
            
        # Simply multiply all learning rates by ratio
        return [lr for _ in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class CustomOneCycleLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3,
                 div_factor=25., final_div_factor=1e4,
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95):
        # Learning rate parameters
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        # Momentum parameters
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        
        # Initialize momentum values if cycling momentum
        if self.cycle_momentum:
            if 'momentum' not in optimizer.defaults and 'betas' not in optimizer.defaults:
                raise ValueError('Optimizer must support momentum with `momentum` option'
                               ' or `betas` option')
            self.use_beta1 = 'betas' in optimizer.defaults
            
        super().__init__(optimizer)

    def get_lr(self):
        import math  # 使用math.pi代替torch.pi
        
        # Calculate where in the cycle we are
        if self.last_epoch <= self.step_size_up:
            # We're in the ramp-up phase
            r = float(self.last_epoch) / float(self.step_size_up)
            # Use cosine function for smooth increase
            cos_out = (1 + math.cos(math.pi + r * math.pi)) / 2
            ratio = 1/self.div_factor + (1 - 1/self.div_factor) * cos_out
        else:
            # We're in the ramp-down phase
            r = float(self.last_epoch - self.step_size_up) / float(self.step_size_down)
            # Use cosine function for smooth decrease
            cos_out = (1 + math.cos(r * math.pi)) / 2
            ratio = cos_out * (1 - 1/(self.div_factor * self.final_div_factor))

        return [base_lr * ratio for base_lr in self.base_lrs]

    def step(self, epoch=None):
        # Update learning rate
        super().step(epoch)
        
        # Update momentum if cycling
        if self.cycle_momentum:
            import math
            
            if self.last_epoch <= self.step_size_up:
                r = float(self.last_epoch) / float(self.step_size_up)
                cos_out = (1 + math.cos(math.pi + r * math.pi)) / 2
                momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * cos_out
            else:
                r = float(self.last_epoch - self.step_size_up) / float(self.step_size_down)
                cos_out = (1 + math.cos(r * math.pi)) / 2
                momentum = self.base_momentum + (self.max_momentum - self.base_momentum) * cos_out
                
            # Update momentum in optimizer
            for pg in self.optimizer.param_groups:
                if self.use_beta1:
                    # For Adam-like optimizers
                    pg['betas'] = (momentum, pg['betas'][1])
                else:
                    # For SGD-like optimizers
                    pg['momentum'] = momentum