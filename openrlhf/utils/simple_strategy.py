#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Simple training strategy for single-device training (e.g., MPS, CPU).
This is a lightweight alternative to DeepspeedStrategy for debugging and local development.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class SimpleStrategy:
    """
    Simple training strategy for single-device training without DeepSpeed.
    """

    def __init__(
        self,
        seed: int = 42,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        args=None,
    ):
        self.seed = seed
        self.micro_train_batch_size = micro_train_batch_size
        self.train_batch_size = train_batch_size
        self.args = args
        self.accumulated_gradient = max(1, train_batch_size // micro_train_batch_size)
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
    
    def setup_distributed(self):
        """No-op for single device training."""
        pass
    
    def is_rank_0(self):
        """Always return True for single device."""
        return True
    
    def get_rank(self):
        """Always return 0 for single device."""
        return 0
    
    def get_world_size(self):
        """Always return 1 for single device."""
        return 1
    
    def print(self, *args, **kwargs):
        """Print only on rank 0 (always true for single device)."""
        print(*args, **kwargs)
    
    def setup_dataloader(
        self,
        dataset,
        batch_size,
        pin_memory=False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    ):
        """Create a simple DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            num_workers=0,  # Use 0 workers for simplicity
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
        )
    
    def prepare_model(self, model):
        """Move model to device."""
        return model.to(self.device)
    
    def prepare(self, *models_or_model_optim_pairs):
        """Prepare models/optimizers for training."""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                # (model, optimizer, scheduler) tuple
                model, optimizer, scheduler = arg
                if model is not None:
                    model = model.to(self.device)
                ret.append((model, optimizer, scheduler))
            else:
                # Just a model
                ret.append(arg.to(self.device))
        
        return ret[0] if len(ret) == 1 else ret
    
    def create_optimizer(self, model, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0, encoder_lr=None, decoder_lr=None):
        """Create a simple AdamW optimizer with optional separate encoder/decoder learning rates."""
        # If separate LRs are specified, create parameter groups
        if encoder_lr is not None or decoder_lr is not None:
            encoder_lr = encoder_lr if encoder_lr is not None else lr
            decoder_lr = decoder_lr if decoder_lr is not None else lr
            
            param_groups = []
            encoder_params = []
            decoder_params = []
            other_params = []
            
            # Separate parameters by adapter type
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'encoder_adapter' in name:
                    encoder_params.append(param)
                elif 'decoder_adapter' in name:
                    decoder_params.append(param)
                else:
                    other_params.append(param)
            
            if encoder_params:
                param_groups.append({'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'})
            if decoder_params:
                param_groups.append({'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'})
            if other_params:
                param_groups.append({'params': other_params, 'lr': lr, 'name': 'other'})
            
            return optim.AdamW(
                param_groups,
                betas=betas,
                weight_decay=weight_decay,
            )
        else:
            # Use single learning rate for all parameters
            return optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
    
    def backward(self, loss, model, optimizer):
        """Simple backward pass."""
        loss.backward()
    
    def optimizer_step(self, optimizer, model, scheduler=None):
        """Simple optimizer step with optional scheduler."""
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    
    def save_model(self, model, tokenizer, path, **kwargs):
        """Save model and tokenizer."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save model
        if hasattr(model, "module"):
            model_to_save = model.module
        else:
            model_to_save = model
        
        # For CLaRa model, use its save_pretrained method
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(path)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))
        
        # Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
    
    def load_ckpt(self, model, path):
        """Load checkpoint - returns (path, states dict)."""
        import os
        if not os.path.exists(path):
            return None, {}
        
        # For now, return empty states - can be extended later
        return path, {}
    
    def all_reduce(self, tensor, op="mean"):
        """No-op for single device."""
        return tensor
    
    def all_gather(self, tensor):
        """No-op for single device."""
        return [tensor]
    
    def barrier(self):
        """No-op for single device."""
        pass
