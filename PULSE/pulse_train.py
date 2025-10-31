import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, BertModel, BertConfig, BertTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import itertools 
from sklearn.preprocessing import LabelEncoder
from itertools import chain
import os
import argparse

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, llada_tokenizer,bert_tokenizer, model_type, max_length=4096):
        self.dataset = dataset
        self.llada_tokenizer = llada_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
        self.model_type = model_type

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        hatespeech = sample['Hatespeech']
        counterspeech = sample["Counterspeech"]
        leader = sample['Style']
        target_label = sample['target_label']

        if leader == "Gandhi":
            leader_label = 0
        else:
            leader_label = 1

        
        if self.model_type=='sslcl':
            prompt_text = ""
            full_text = f"{self.llada_tokenizer.bos_token}{counterspeech}{self.llada_tokenizer.eos_token}"
        elif self.model_type=='sccg':
            prompt_text = f"Generate an effective counterspeech for the hatespeech: {hatespeech} in the style of {leader}. Counterspeech:"
            response_text = f" {counterspeech}"
            full_text = f"{self.llada_tokenizer.bos_token}{prompt_text}{response_text}{self.llada_tokenizer.eos_token}"
        
        llada_encoding = self.llada_tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        llada_input_ids = llada_encoding["input_ids"].squeeze(0)

        
        # Tokenize for style encoder (BERT)
        bert_encoding = self.bert_tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        bert_input_ids = bert_encoding["input_ids"].squeeze(0)
        
        # To calculate prompt length, tokenize only the prompt part
        prompt_encoding = self.llada_tokenizer(f"{self.llada_tokenizer.bos_token}{prompt_text}", add_special_tokens=False, return_tensors="pt")
        prompt_length = prompt_encoding["input_ids"].shape[1]
        
        return {
            "llada_input_ids": llada_input_ids,
            "bert_input_ids": bert_input_ids,
            "prompt_lengths": torch.tensor(prompt_length, dtype=torch.long),
            "leader_label": leader_label,
            "target_label": torch.tensor(target_label, dtype=torch.long)
        }

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs, style_labels=None):
        
        if inputs.dim() > 2:
             inputs = inputs.squeeze(-1).squeeze(-1)
        
        # forced lcv selection
        if style_labels is not None:
            encoding_indices = style_labels
        else:
            distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                        + torch.sum(self.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            encoding_indices = torch.argmin(distances, dim=1)

        quantized = self.embedding(encoding_indices)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized, encoding_indices

class PerStyM(nn.Module):
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*hidden_size, 2*hidden_size),
                nn.GELU(),
                nn.LayerNorm(2*hidden_size)
            ) for _ in range(num_layers)
        ])
        self.semantic_gate = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.style_gate = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, zs, ef):
        ef = ef.unsqueeze(1).expand_as(zs)
        combined = torch.cat([zs, ef], dim=-1)
        for layer in self.transform_layers:
            transformed = layer(combined)
            combined = combined + transformed # Residual connection
        
        μ_sem = self.semantic_gate(combined)
        μ_sty = self.style_gate(combined)

        # blending transformed semantics.
        z_sem = (1 - μ_sem) * zs + μ_sem * combined[..., :zs.size(-1)]
        z_sty = (1 - μ_sty) * zs + μ_sty * ef
        
        # The final fusion combines semantic and style-influenced representations.
        fused_output = z_sem + z_sty
        return fused_output

class LLaDA_SSLCL:
    def __init__(self, model_name, train_csv, eval_csv, device='cuda', use_vq=True, learning_rate=2.5e-5, model_type='sslcl', latent_code_vector_dim = 768):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_vq = use_vq
        self.model_type = model_type
        self.latent_code_vector_dim = latent_code_vector_dim
        self.model, self.tokenizer = self._load_model_tokenizer(model_name)
        self.style_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.style_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
        # Load the style encoder model
        self.style_encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        # Resize the model's embeddings to match the new tokenizer size
        self.style_encoder.resize_token_embeddings(len(self.style_tokenizer))
        
        self.style_encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.style_encoder.train()
        # Set mask_id after tokenizer is loaded
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.dataset = self._load_dataset(train_csv)
        self.val_dataset = self._load_dataset(eval_csv)
        print("Vocab size:", len(self.tokenizer))
        print("Hidden size:", self.model.config.hidden_size)
        
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=2, shuffle=True)
        
        
        if(self.latent_code_vector_dim!=768):
            self.pre_quant_projection = nn.Linear(self.style_encoder.config.hidden_size, self.latent_code_vector_dim).to(self.device)
        
        self.style_projection = nn.Linear(self.latent_code_vector_dim, self.model.config.hidden_size).to(self.device)
        
        
        if self.use_vq:
            self.style_quantizer = VectorQuantizer(num_embeddings=2, embedding_dim=self.latent_code_vector_dim).to(self.device)
            self.fusion = PerStyM(self.model.config.hidden_size).to(self.device)
            self.output_layer = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size).to(self.device)
        
        self._configure_optimizer(learning_rate)

    def _load_model_tokenizer(self, model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        model = get_peft_model(model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens = {
            "pad_token": "[PAD]", # Explicitly set pad token
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "additional_special_tokens": ["<start_id>", "<end_id>", "<eot_id>", "[MASK]"]
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        # Set pad token id in model config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(self.device)
        return model, tokenizer

    def _configure_optimizer(self, learning_rate):
        params_to_optimize = self.model.parameters()
        if self.use_vq:
            # Combine parameters from all modules for the optimizer
            params_to_optimize = itertools.chain(
                self.model.parameters(), 
                self.style_encoder.parameters(),
                self.style_quantizer.parameters(), 
                self.style_projection.parameters(),
                self.fusion.parameters(),
                self.output_layer.parameters()
            )
        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    def _load_dataset(self, train_csv):
        df_train = pd.read_csv(train_csv)
        target_encoder = LabelEncoder()
        df_train['target_label'] = target_encoder.fit_transform(df_train['Target'])
        raw_dataset = Dataset.from_pandas(df_train)
        return HateSpeechDataset(raw_dataset, self.tokenizer, self.style_tokenizer, self.model_type, max_length=512)

    @staticmethod
    def forward_process(input_ids, mask_id, eps=1e-3):
        b, l = input_ids.shape
        device = input_ids.device
        t = torch.rand(b, device=device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=device) < p_mask
        mask_id_tensor = torch.tensor(mask_id, dtype=torch.long, device=device)
        noisy_batch = torch.where(masked_indices, mask_id_tensor, input_ids)
        
        return noisy_batch, masked_indices, p_mask

    def train(self, epochs=3):
        self.model.train()
        if self.use_vq:
            self.style_quantizer.train()
            self.fusion.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", leave=False)
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                llada_input_ids = batch["llada_input_ids"].to(self.device)
                bert_input_ids = batch["bert_input_ids"].to(self.device)
                prompt_lengths = batch["prompt_lengths"].to(self.device)
                leader_labels = batch["leader_label"].to(self.device)
                
                # Apply diffusion-like masking process
                noisy_batch, masked_indices, p_mask = self.forward_process(llada_input_ids, mask_id=self.mask_id)
                
                # Ensure the prompt part remains unmasked
                token_positions = torch.arange(noisy_batch.shape[1], device=self.device).expand_as(noisy_batch)
                prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
                noisy_batch[prompt_mask] = llada_input_ids[prompt_mask]
                
                # Re-calculate masked_indices after preserving the prompt
                masked_indices = (noisy_batch == self.mask_id) & (~prompt_mask)
                
                if masked_indices.sum() == 0:
                    continue
                
                # Initialize vq_loss to 0.0
                vq_loss = 0.0
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Request hidden states for fusion module
                    outputs = self.model(input_ids=noisy_batch, output_hidden_states=True)
                    logits = outputs.logits
                    sequence_output = outputs.hidden_states[-1] # Last hidden state
                    # print("sequence output: ", sequence_output.shape)
                    if self.use_vq:
                        style_encoder_outputs = self.style_encoder(input_ids=bert_input_ids)
                        style_hidden_states = style_encoder_outputs.last_hidden_state
                        style_embedding_mean = torch.mean(style_hidden_states, dim=1)
                        if(self.latent_code_vector_dim!=768):
                            style_embedding_mean = self.pre_quant_projection(style_embedding_mean)
                        
                        vq_loss, quantized_style_vec, _ = self.style_quantizer(style_embedding_mean, leader_labels)
                        style_lcv = self.style_projection(quantized_style_vec)
                        fused_hidden_states = self.fusion(sequence_output, style_lcv)
                        logits = self.output_layer(fused_hidden_states)

                    # Loss is calculated only on the masked (non-prompt) tokens
                    ce_loss = F.cross_entropy(
                        logits.view(-1, self.model.config.vocab_size),
                        llada_input_ids.view(-1),
                        ignore_index=self.tokenizer.pad_token_id,
                        reduction='none'
                    )
                    
                    # Apply mask and normalize
                    masked_loss = ce_loss[masked_indices.view(-1)]
                    final_loss = masked_loss.mean() + vq_loss

                final_loss.backward()
                self.optimizer.step()
                
                total_loss += final_loss.item()
                progress_bar.set_postfix({"Loss": f"{final_loss.item():.4f}"})

            print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(self.dataloader):.4f}")
            
            checkpoint_dir = f'checkpoints/{self.model_type}/{self.latent_code_vector_dim}'
            checkpoint_path = os.path.join(checkpoint_dir, f'sslcl_epoch_{epoch+1}.pth')

            # Create a dictionary with all necessary states[3][5]
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'style_quantizer_state_dict': self.style_quantizer.state_dict(),
                'style_projection_state_dict': self.style_projection.state_dict(),
                'fusion_state_dict': self.fusion.state_dict()
                
            }

            # Save the checkpoint dictionary to a file
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    def evaluate(self, val_dataloader):
        self.model.eval()
        if self.use_vq:
            self.style_quantizer.eval()
            self.fusion.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_dataloader:
                llada_input_ids = batch["llada_input_ids"].to(self.device)
                bert_input_ids = batch["bert_input_ids"].to(self.device)
                prompt_lengths = batch["prompt_lengths"].to(self.device)
                leader_labels = batch["leader_label"].to(self.device)
                noisy_batch, masked_indices, p_mask = self.forward_process(llada_input_ids, mask_id=self.mask_id)
                token_positions = torch.arange(noisy_batch.shape[1], device=self.device).expand_as(noisy_batch)
                prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
                noisy_batch[prompt_mask] = llada_input_ids[prompt_mask]
                masked_indices = (noisy_batch == self.mask_id) & (~prompt_mask)
                if masked_indices.sum() == 0:
                    continue
                vq_loss = 0.0
                outputs = self.model(input_ids=noisy_batch, output_hidden_states=True)
                logits = outputs.logits
                sequence_output = outputs.hidden_states[-1]
                if self.use_vq:
                    style_encoder_outputs = self.style_encoder(input_ids=bert_input_ids)
                    style_hidden_states = style_encoder_outputs.last_hidden_state
                    style_embedding_mean = torch.mean(style_hidden_states, dim=1)
                    if(self.latent_code_vector_dim!=768):
                        style_embedding_mean = self.pre_quant_projection(style_embedding_mean)
                    vq_loss, quantized_style_vec, _ = self.style_quantizer(style_embedding_mean, leader_labels)
                    style_lcv = self.style_projection(quantized_style_vec)
                    fused_hidden_states = self.fusion(sequence_output, style_lcv)
                    logits = self.output_layer(fused_hidden_states)
                ce_loss = F.cross_entropy(
                    logits.view(-1, self.model.config.vocab_size),
                    llada_input_ids.view(-1),
                    ignore_index=self.tokenizer.pad_token_id,
                    reduction='none'
                )
                masked_loss = ce_loss[masked_indices.view(-1)]
                final_loss = masked_loss.mean() + vq_loss
                total_loss += final_loss.item()
                count += 1
        print(f"Validation Loss: {total_loss / count:.4f}")
        self.model.train()
        if self.use_vq:
            self.style_quantizer.train()
            self.fusion.train()

class LLaDA_SCCG:
    def __init__(self, model_name, train_csv, sslcl_model, model_type = 'sccg', device='cuda', use_vq=True, learning_rate=2.5e-5, latent_code_vector_dim=768):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_vq = use_vq
        self.model_type = model_type
        self.latent_code_vector_dim = latent_code_vector_dim
        self.model, self.tokenizer = self._load_model_tokenizer(model_name)
        self.style_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.style_tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']})
        # Load the style encoder model
        self.style_encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        #  Resize the model's embeddings to match the new tokenizer size
        self.style_encoder.resize_token_embeddings(len(self.style_tokenizer))
        
        self.style_encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        for param in self.style_encoder.parameters():
            param.requires_grad = False
        self.style_encoder.eval()
        #  Set mask_id after tokenizer is loaded
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.dataset = self._load_dataset(train_csv)
        print("Vocab size:", len(self.tokenizer))
        print("Hidden size:", self.model.config.hidden_size)
        
        self.dataloader = DataLoader(self.dataset, batch_size=5, shuffle=True)
        if(self.latent_code_vector_dim!=768):
            self.pre_quant_projection = nn.Linear(self.style_encoder.config.hidden_size, self.latent_code_vector_dim).to(self.device)
        self.style_projection = nn.Linear(self.latent_code_vector_dim, self.model.config.hidden_size).to(self.device)
        
        # added target embedding and target classifer
        self.target_embedding = nn.Embedding(9, 4096)  # target categories
        self.target_classifier = nn.Sequential(
            nn.Linear(4096, 4096), nn.LayerNorm(4096), nn.GELU(), nn.Linear(4096, 9)
        ).to(self.device)
        
        self.model.load_state_dict(sslcl_model['model_state_dict'], strict=False)
        
        if self.use_vq:
            self.style_quantizer = VectorQuantizer(num_embeddings=2, embedding_dim=self.latent_code_vector_dim).to(self.device)
            self.style_quantizer.load_state_dict(sslcl_model['style_quantizer_state_dict'], strict = False)
            for param in chain(self.style_quantizer.parameters()):
                param.requires_grad = False
            self.fusion = PerStyM(self.model.config.hidden_size).to(self.device)
            self.fusion.load_state_dict(sslcl_model['fusion_state_dict'], strict = False)
            self.output_layer = nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size).to(self.device)
        
        self._configure_optimizer(learning_rate)
        
    def _load_model_tokenizer(self, model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
        model = get_peft_model(model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        special_tokens = {
            "pad_token": "[PAD]", # Explicitly set pad token
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "additional_special_tokens": ["<start_id>", "<end_id>", "<eot_id>", "[MASK]"]
        }
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        # Set pad token id in model config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(self.device)
        return model, tokenizer

    def _configure_optimizer(self, learning_rate):
        params_to_optimize = self.model.parameters()
        if self.use_vq:
            # Combine parameters from all modules for the optimizer
            params_to_optimize = itertools.chain(
                self.model.parameters(), 
                # self.style_encoder.parameters(),
                # self.style_quantizer.parameters(), 
                self.style_projection.parameters(),
                self.fusion.parameters(),
                self.output_layer.parameters()
            )
        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate)

    def _load_dataset(self, train_csv):
        df_train = pd.read_csv(train_csv)
        target_encoder = LabelEncoder()
        df_train['target_label'] = target_encoder.fit_transform(df_train['Target'])
        raw_dataset = Dataset.from_pandas(df_train)
        return HateSpeechDataset(raw_dataset, self.tokenizer, self.style_tokenizer, self.model_type, max_length=512)

    @staticmethod
    def forward_process(input_ids, mask_id, eps=1e-3):
        b, l = input_ids.shape
        device = input_ids.device
        t = torch.rand(b, device=device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=device) < p_mask
        mask_id_tensor = torch.tensor(mask_id, dtype=torch.long, device=device)
        noisy_batch = torch.where(masked_indices, mask_id_tensor, input_ids)
        
        return noisy_batch, masked_indices, p_mask

    def train(self, epochs=3):
        self.model.train()
        if self.use_vq:
            self.style_quantizer.train()
            self.fusion.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", leave=False)
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                llada_input_ids = batch["llada_input_ids"].to(self.device)
                bert_input_ids = batch["bert_input_ids"].to(self.device)
                prompt_lengths = batch["prompt_lengths"].to(self.device)
                leader_labels = batch["leader_label"].to(self.device)
                targets = batch["target_label"].to(self.device)
                
                # Apply diffusion-like masking process
                noisy_batch, masked_indices, p_mask = self.forward_process(llada_input_ids, mask_id=self.mask_id)
                
                # Ensure the prompt part remains unmasked
                token_positions = torch.arange(noisy_batch.shape[1], device=self.device).expand_as(noisy_batch)
                prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
                noisy_batch[prompt_mask] = llada_input_ids[prompt_mask]
                
                # Re-calculate masked_indices after preserving the prompt
                masked_indices = (noisy_batch == self.mask_id) & (~prompt_mask)
                
                if masked_indices.sum() == 0:
                    continue
                
                # Initialize vq_loss to 0.0
                vq_loss = 0.0
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Request hidden states for fusion module
                    outputs = self.model(input_ids=noisy_batch, output_hidden_states=True)
                    logits = outputs.logits
                    sequence_output = outputs.hidden_states[-1] # Last hidden state
                    print("sequence output: ", sequence_output.shape)
                    if self.use_vq:
                        style_encoder_outputs = self.style_encoder(input_ids=bert_input_ids)
                        style_hidden_states = style_encoder_outputs.last_hidden_state
                        style_embedding_mean = torch.mean(style_hidden_states, dim=1)
                        
                        if(self.latent_code_vector_dim!=768):
                            style_embedding_mean = self.pre_quant_projection(style_embedding_mean)
                        vq_loss, quantized_style_vec, _ = self.style_quantizer(style_embedding_mean, leader_labels)
                        style_lcv = self.style_projection(quantized_style_vec)
                        fused_hidden_states = self.fusion(sequence_output, style_lcv)
                        logits = self.output_layer(fused_hidden_states)

                    # Loss is calculated only on the masked (non-prompt) tokens
                    ce_loss = F.cross_entropy(
                        logits.view(-1, self.model.config.vocab_size),
                        llada_input_ids.view(-1),
                        ignore_index=self.tokenizer.pad_token_id,
                        reduction='none'
                    )
                    
                    # Auxiliary target loss
                    target_logits = self.target_classifier(sequence_output.mean(dim=1))
                    
                    
                    # Target-aware auxiliary loss
                    target_loss = F.cross_entropy(
                        target_logits,
                        targets
                    )
                    
                    # Apply mask and normalize
                    masked_loss = ce_loss[masked_indices.view(-1)]
                    final_loss = masked_loss.mean() + target_loss

                final_loss.backward()
                self.optimizer.step()
                
                total_loss += final_loss.item()
                progress_bar.set_postfix({"Loss": f"{final_loss.item():.4f}"})

            print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(self.dataloader):.4f}")
            ck_dir = f'checkpoints/{self.model_type}/{self.latent_code_vector_dim}'
            if(epoch>5):
                self.save_checkpoint(epoch, ck_dir)
    # Add this method inside the LLaDA_SCCG class
    def save_checkpoint(self, epoch, checkpoint_dir):
        """Saves the model's state to a checkpoint file."""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'sccg_epoch_{epoch}.pth')

        # Create a dictionary with all the necessary state dictionaries
        # We save only the components that are trained in this stage.
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'style_projection_state_dict': self.style_projection.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'output_layer_state_dict': self.output_layer.state_dict(),
            'target_classifier_state_dict': self.target_classifier.state_dict()
        }

        # Save the checkpoint dictionary to a file
        torch.save(checkpoint, checkpoint_path)
        print(f"SCCG checkpoint saved to {checkpoint_path}")

def main(args):
    model_type = args.model_type
    latent_code_vector_dim = args.latent_code_vector_dim
    epochs = args.epochs

    if model_type == 'sslcl':
        trainer = LLaDA_SSLCL(
            model_name=args.model_name,
            train_csv=args.train_csv,
            eval_csv = args.val_csv,
            use_vq=args.use_vq,
            learning_rate=args.learning_rate,
            latent_code_vector_dim=latent_code_vector_dim,
        )
    elif model_type == 'sccg':
        SSLCL_MODEL_PATH = f"checkpoints/sslcl/{latent_code_vector_dim}/sslcl_epoch_4.pth"
        device = "cuda"
        checkpoint = torch.load(SSLCL_MODEL_PATH, map_location=device)    
        trainer = LLaDA_SCCG(
            model_name=args.model_name,
            train_csv=args.train_csv,
            sslcl_model=checkpoint,
            use_vq=args.use_vq,
            learning_rate=args.learning_rate,
            latent_code_vector_dim=latent_code_vector_dim,
        )

    print("------Starting model training------")
    trainer.train(epochs=epochs)
    print("------Finish model training------")
    # checkpoint_dir = f'results/{model_type}'
    # torch.save(trainer.state_dict(), os.path.join(checkpoint_dir, f'checkpoint.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaDA model using SSLCL or SCCG")
    parser.add_argument('--model_type', type=str, default='sslcl', choices=['sslcl', 'sccg'], help='Type of model to train')
    parser.add_argument('--latent_code_vector_dim', type=int, default=768, help='Latent code vector dimension')
    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct', help='Model path')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--use_vq', action='store_true', default=True, help='Enable style training (VQ)')
    parser.add_argument('--learning_rate', type=float, default=2.5e-5, help='Learning rate')
    args = parser.parse_args()
    main(args)
