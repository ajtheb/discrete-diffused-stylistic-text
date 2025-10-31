
from .loss import step_fn
import torch.optim as optim
from itertools import chain
import os
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from transformers import BartTokenizer, BartModel

class Trainer:
    def __init__(self, run, model, graph, noise, config, style_quantizer=None, eval_callback=None, sample_callback=None, device='cuda', checkpoint_dir='checkpoints'):
        self.graph = graph
        self.model = model
        self.noise = noise
        self.config = config
        self.style_quantizer = style_quantizer
        self.eval_callback = eval_callback
        self.sample_callback = sample_callback
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run = run

        self.use = config['model'].get('use_style_embedding', False)
        self.style_dim = config['model'].get('style_dim', 768)
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, dataset):
        cfg = self.config
        optim_params = [
            self.model.parameters(),
            self.noise.parameters()
        ]
        # add parameters for style_quantizers for optimizer
        if self.use:
            optim_params.extend([
                self.style_quantizer.parameters()
            ])
        # build optimization state
        optimizer = optimizer = optim.AdamW(
            chain(*optim_params),
            lr=cfg['optim']['lr'],
            betas=(cfg['optim']['beta1'],
                cfg['optim']['beta2']),
            eps=cfg['optim']['eps'],
            weight_decay=cfg['optim']['weight_decay']
        )

        state = dict(
            optimizer=optimizer,
            model=self.model,
            noise=self.noise,
            graph=self.graph,
            step=0,
            style_quantizer=self.style_quantizer
        )

        n_epochs = cfg['training']['n_epochs']
        for e in tqdm(range(n_epochs)):
            print(f"***************Epoch {e}******************************")
            for batch in dataset:
                self.step(state, batch)

    def step(self, state, batch):
        cfg = self.config
        step = state['step']
        
        # change
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        #change
        style_labels = batch['style'].to(self.device)
        target_labels = batch['target'].to(self.device)
        
        batch_input_ids = batch['input_ids']
        
        loss, p_batch = step_fn(cfg, state, batch_input_ids, self.use, train=True, style_labels=style_labels, target_labels=target_labels)

        #self.run.track(loss.item(), name='loss', step=state['step'], context={ "subset":"train" })

        # flag to see if there was movement ie a full batch got computed
        if step % cfg['training']['log_freq'] == 0:
            print("step: %d, training_loss: %.5e" % (step, loss.item()))
            # from transformers import GPT2TokenizerFast
            # tokenizer = GPT2TokenizerFast.from_pretrained('/home/models/gpt2')
            # tokenizer.pad_token = '<PAD>'
            # decoded_samples = tokenizer.batch_decode(p_batch[:3].cpu())
            
            # print(f"\nStep {step} - Perturbed Samples:")
            # for i, text in enumerate(decoded_samples):
            #     print(f"Sample {i+1}: {text[:150]}...")

        if step % cfg['training']['eval_freq'] == 0:
            if self.eval_callback is not None:
                self.eval_callback(state)

        if step % cfg['training']['snapshot_freq'] == 0:
            torch.save(state['model'].state_dict(), os.path.join(self.checkpoint_dir, f'checkpoint_{step}.pth'))

        if step > 0 and step % cfg['training']['snapshot_freq'] == 0:
            # Generate and save samples
            if self.sample_callback is not None:
                self.sample_callback(state)
