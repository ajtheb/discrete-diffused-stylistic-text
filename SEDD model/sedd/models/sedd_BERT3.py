import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
import math
import numpy as np
from itertools import chain
#CHANGE
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs, style_labels=None):
        """
        Args:
            inputs: Encoder outputs [batch_size, embedding_dim, 1, 1]
            style_labels: List/Tensor of style indices [batch_size]
        """
        # Convert to (batch_size, embedding_dim)
        inputs = inputs.squeeze(-1).squeeze(-1)
        
        if style_labels is not None:
            # Forced alignment mode (training)
            # Use ground-truth style indices
            encoding_indices = style_labels
            # print("here")
        else:
            # Inference mode: nearest neighbor lookup
            distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                        + torch.sum(self.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(inputs, self.embedding.weight.t()))
            encoding_indices = torch.argmin(distances, dim=1)

        # Quantize using selected indices
        quantized = self.embedding(encoding_indices)
        
        # Loss calculations
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Reshape to match original input dimensions
        quantized = quantized.unsqueeze(-1).unsqueeze(-1)
        
        return loss, quantized, encoding_indices
    
# TimestepEmbedder: Embeds scalar timesteps into vector representations
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        # MLP to process frequency embeddings
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    # Static method to create sinusoidal timestep embeddings
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    # Forward pass: create and process timestep embeddings
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

#CHANGE
class PerFuMe(nn.Module):
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*hidden_size, 2*hidden_size),
                nn.GELU(),
                nn.LayerNorm(2*hidden_size)
            ) for _ in range(num_layers)
        ])
        
        # Adaptive gates
        self.semantic_gate = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.style_gate = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, zs, ef):
        """
        zs: semantic sequence output [batch_size, seq_len, hidden_size]
        ef: style codebook embeddings [batch_size, hidden_size]
        """
        # Expand style embeddings to match sequence length
        ef = ef.unsqueeze(1).expand_as(zs)  # [batch_size, seq_len, hidden_size]
        
        # Persistent fusion with residual connections
        combined = torch.cat([zs, ef], dim=-1)
        for layer in self.transform_layers:
            transformed = layer(combined)
            combined = combined + transformed  # Residual connection
            
        # Adaptive gating mechanism
        μ_s = self.semantic_gate(combined)  # Semantic gate [batch_size, seq_len, hidden_size]
        μ_i = self.style_gate(combined)     # Style gate [batch_size, seq_len, hidden_size]
        
        # Gate-controlled fusion
        z_sem = (1 - μ_s) * zs + μ_s * zs   # Semantic persistence
        z_int = (1 - μ_i) * zs + μ_i * ef   # Style integration
        
        # Final combined representation
        fused_output = z_sem * z_int + combined[..., :zs.size(-1)]
        return fused_output

#CHANGE
class PerFuMe_Target(nn.Module):
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3*hidden_size, 3*hidden_size),
                nn.GELU(),
                nn.LayerNorm(3*hidden_size)
            ) for _ in range(num_layers)
        ])
        
        # Adaptive gates
        self.semantic_gate = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.style_gate = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.target_gate = nn.Sequential(
            nn.Linear(3*hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.final_proj = nn.Linear(3*hidden_size, hidden_size)

    def forward(self, zs, ef, et):
        """
        zs: semantic sequence output [batch_size, seq_len, hidden_size]
        ef: style codebook embeddings [batch_size, hidden_size]
        """
        # Expand style embeddings to match sequence length
        ef = ef.unsqueeze(1).expand_as(zs)  # [batch_size, seq_len, hidden_size]
        et = et.expand_as(zs)
        # Persistent fusion with residual connections
        combined = torch.cat([zs, ef, et], dim=-1)
        for layer in self.transform_layers:
            transformed = layer(combined)
            combined = combined + transformed  # Residual connection
            
        # Adaptive gating mechanism
        μ_s = self.semantic_gate(combined)  # Semantic gate [batch_size, seq_len, hidden_size]
        μ_i = self.style_gate(combined)     # Style gate [batch_size, seq_len, hidden_size]
        μ_t = self.target_gate(combined)
        
        # Gate-controlled fusion
        z_sem = (1 - μ_s) * zs + μ_s * combined[..., :zs.size(-1)]
        z_int = (1 - μ_i) * zs + μ_i * ef
        z_tgt = (1 - μ_t) * zs + μ_t * et
        fused_output = torch.cat([z_sem, z_int, z_tgt], dim=-1)
        fused_output = self.final_proj(fused_output)

        return fused_output


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, semantic_feats, style_feats):
        Q = self.query(semantic_feats)
        K = self.key(style_feats)
        V = self.value(style_feats)
        attn_weights = torch.softmax(Q @ K.transpose(-2,-1) / np.sqrt(K.size(-1)), dim=-1)
        return attn_weights @ V
    
# SEDD_BERT: Main model class combining BERT with SEDD components
class SEDD_BERT_CLIME(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        # Initialize BERT model
        bert_config = BertConfig.from_pretrained('/home/models/bert-base-uncased')
        bert_config.vocab_size += 1  # Add 1 for absorbing state in the Bert_config
        self.bert = BertModel.from_pretrained('/home/models/bert-base-uncased',
                                                config=bert_config, 
                                                ignore_mismatched_sizes=True)
        #CHANGE
        # Personality Encoding BERT 
        self.personality_bert = BertModel.from_pretrained('/home/models/bert-base-uncased',
                                                           config=bert_config, 
                                                            ignore_mismatched_sizes=True)
        # self.personality_pooler = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.Tanh()
        # )
        
        # Timestamp embedder
        self.timestamp_embed = TimestepEmbedder(bert_config.hidden_size)

        # Output layer
        self.output_layer = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()

        self.use_style_embedding = config['model'].get('use_style_embedding', True)
        # Add vector quantizer for style codebook
        if self.use_style_embedding:
            self.style_quantizer = VectorQuantizer(num_embeddings=2, embedding_dim=768)
            
            self.fusion = PerFuMe(768) # bert vq
            # self.fusion = EnhancedPerFuMe(768)
        else:
            self.style_quantizer = None
            
        
            
    # Forward pass of the SEDD_BERT model
    def forward(self, input_ids, sigma, style_codebook=None, style_labels = None, target_labels = None):
        # Personality Encoding (Phase 1 CLIME) #CHANGE
        personality_outputs = self.personality_bert(input_ids)
        personality_outputs = personality_outputs.last_hidden_state
        # pooled_personality = self.personality_pooler(
        #     torch.mean(personality_outputs.last_hidden_state, dim=1)
        # )
        # Embed timestamps
        timestamp_embeddings = self.timestamp_embed(sigma)

        # BERT forward pass
        outputs = self.bert(input_ids, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        # Add timestamp embeddings to BERT output
        sequence_output += timestamp_embeddings.unsqueeze(1)

        vq_loss = 0.0
        # vq_flag is 1
        if self.use_style_embedding:
            # print("sequence_output", sequence_output.shape)
            personality_outputs_mean = torch.mean(personality_outputs, dim=1)
            # print("sequence output", sequence_output.shape)    
            # get codebook
            vq_loss,style_codebook, _ = self.style_quantizer(personality_outputs_mean, style_labels=style_labels)
            style_codebook = style_codebook.squeeze(-1).squeeze(-1)
            
            # print("style_codeboook", style_codebook.shape)
            
            # fusion of input and codebook
            sequence_output = self.fusion(sequence_output, style_codebook)
            # print("after fusion:", x.shape)
            
        # Add retrieved context via cross-attention
        # if context:
        #     sequence_output = self.cross_attention(sequence_output, context)
            
        # print("After adding sequence output:  ", sequence_output.shape)
        # Final output layer
        logits = self.output_layer(sequence_output)

        # Apply sigma correction for diffusion process
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(logits.dtype)[:, None, None]
        logits = logits - esigm1_log - torch.log(torch.tensor(logits.shape[-1] - 1, dtype=logits.dtype, device=logits.device))

        # Zero out logits for absorbing state
        logits = torch.scatter(logits, -1, input_ids.unsqueeze(-1), torch.zeros_like(logits[..., :1]))

        return logits, vq_loss


class SEDD_BERT_COGENT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, clime_model):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        # Freeze CLIME components
        self.personality_bert = clime_model.personality_bert
        self.style_quantizer = clime_model.style_quantizer
        
        print(self.style_quantizer)
        bert_config = BertConfig.from_pretrained('/home/models/bert-base-uncased')
        bert_config.vocab_size += 1
        
        for param in chain(self.personality_bert.parameters(), self.style_quantizer.parameters()):
            param.requires_grad = False
            
        # Reuse main BERT from CLIME as semantic encoder
        self.bert = clime_model.bert
        print("clime", clime_model.bert.config.vocab_size)
        self.bert.resize_token_embeddings(clime_model.bert.config.vocab_size)
        
        # Contextual Mapper (TREAD implementation)
        # self.contextual_mapper = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=768,
        #         nhead=12,
        #         dim_feedforward=3072,
        #         activation='gelu'
        #     ),
        #     num_layers=4
        # )
        # Timestamp embedder
        self.timestamp_embed = TimestepEmbedder(bert_config.hidden_size)

        # Output layer
        self.output_layer = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()
        
        # added target embedding and target classifer(CHANGE)
        self.target_embedding = nn.Embedding(9, 768)  # 9->8 target categories
        self.target_classifier = nn.Sequential(
            nn.Linear(768, 768), nn.LayerNorm(768), nn.GELU(), nn.Linear(768, 9)
        )

        self.use_style_embedding = config['model'].get('use_style_embedding', True)
        # Add vector quantizer for style codebook
        if self.use_style_embedding:
            # self.style_quantizer = VectorQuantizer(num_embeddings=2, embedding_dim=768)
            
            self.fusion = clime_model.fusion # Reuse of weights
            # self.fusion = PerFuMe(768) # new weights
            # self.fusion = CrossModalAttention(768)
            # self.fusion = PerFuMe_Target(768)
        else:
            self.style_quantizer = None
            
        
            
    # Forward pass of the SEDD_BERT_COGENT model
    def forward(self, input_ids, sigma, style_codebook=None, style_labels = None, targets = None):
        # print("in forward")
        # Phase 1: Get hate speech semantics
        # print( input_ids)
        # print("vocab", self.bert.config.vocab_size)
        hate_semantics = self.bert(input_ids).last_hidden_state
        # print("targets" ,targets)
        # Target information incorporation
        target_embeds = self.target_embedding(targets).unsqueeze(1)
        
        # print("hate semantics:",hate_semantics.shape)
        
        # check with add and no add
        hate_semantics += target_embeds
        
        # Contextual mapping (TREAD)
        # mapped_semantics = self.contextual_mapper(hate_semantics)
        
        # Embed timestamps
        timestamp_embeddings = self.timestamp_embed(sigma)
    
        # print("sequence_output", sequence_output.shape)
        mapped_semantics_mean = torch.mean(hate_semantics, dim=1)
        # print("sequence output", sequence_output.shape)    
        with torch.no_grad():
            # print("style_codeboook", style_codebook.shape)
            _,style_codebook, _ = self.style_quantizer(mapped_semantics_mean, style_labels=style_labels)
            style_codebook = style_codebook.squeeze(-1).squeeze(-1)
        # print("hate_semantics", hate_semantics.shape)
        # print("style_codebook", style_codebook.shape)
        # print("target_embeds", target_embeds.shape)
        # fusion of input and codebook
        sequence_output = self.fusion(hate_semantics, style_codebook)
        # Add timestamp embeddings to BERT output
        sequence_output += timestamp_embeddings.unsqueeze(1)
            
        # print("After adding sequence output:  ", sequence_output.shape)
        # Final output layer
        logits = self.output_layer(sequence_output)

        # Apply sigma correction for diffusion process
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(logits.dtype)[:, None, None]
        logits = logits - esigm1_log - torch.log(torch.tensor(logits.shape[-1] - 1, dtype=logits.dtype, device=logits.device))

        # Zero out logits for absorbing state
        logits = torch.scatter(logits, -1, input_ids.unsqueeze(-1), torch.zeros_like(logits[..., :1]))

        # Auxiliary target loss
        target_logits = self.target_classifier(hate_semantics.mean(dim=1))
        
        
        # Target-aware auxiliary loss
        target_loss = F.cross_entropy(
            target_logits,
            targets
        )
        
        # vq_loss+=target_loss
        
        return logits, target_loss

# Score function: Computes the score for the diffusion process
def score_fn(model, x, sigma, use, step,  train=True, sampling=False, style_labels = None, target_labels = None):
    # print("targets", target_labels)
    sigma = sigma.reshape(-1)
    if train:
        model.train()
    else:
        model.eval()

    score, aux_loss = model(x, sigma, use, style_labels, target_labels)

    if sampling:
        return score.exp()
    return score, aux_loss
