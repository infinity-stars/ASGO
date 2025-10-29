import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from omegaconf import OmegaConf

class LlamaModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not isinstance(config, dict):
            config = OmegaConf.to_container(config, resolve=True)
        llama_config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
        for key, value in config.items():
            setattr(llama_config, key, value)
        self.model = LlamaForCausalLM(llama_config)
        print(f"Config of the model: {llama_config}")
        print(f"number of parameters: {self.get_num_params()/1e6}M")

    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.embed_tokens.weight.numel()
        return n_params
    
    def forward(self, idx):
        "Forward pass through the model"
        outputs = self.model(input_ids = idx)
        return outputs.logits

    def crop_block_size(self, block_size):
        "Crop the model to a smaller block size"
        assert block_size <= self.config.block_size, f"block_size {block_size} must be less than {self.config.block_size}"
        self.config.block_size = block_size
        self.model.config.max_position_embeddings = block_size

    @classmethod
    def from_pretrained(cls, model_type, config, override_args=None):
        """Load a pretrained Llama model."""
        # This method can be implemented if you want to load pretrained weights
        # For now, we'll just return a new model
        return cls(config)
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
        # Llama model FLOPs estimation
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 2*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text using the model."""
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx