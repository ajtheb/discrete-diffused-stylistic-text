import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
import math
import numpy as np

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
            print("here")
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
    
    # def forward(self, inputs):
    #     inputs = inputs.permute(0, 2, 3, 1).contiguous()
    #     input_shape = inputs.shape
    #     flat_input = inputs.view(-1, self.embedding.embedding_dim)
    #     distances = torch.sum(flat_input**2, dim=1, keepdim=True) + \
    #                 torch.sum(self.embedding.weight**2, dim=1) - \
    #                 2 * torch.matmul(flat_input, self.embedding.weight.t())
    #     # get indices of closest codebook
    #     encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    #     encodings = torch.zeros(encoding_indices.shape[0], self.embedding.num_embeddings, device=inputs.device)
    #     encodings.scatter_(1, encoding_indices, 1)
    #     # get codebooks
    #     quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
    #     e_latent_loss = F.mse_loss(quantized.detach(), inputs)
    #     q_latent_loss = F.mse_loss(quantized, inputs.detach())
    #     loss = q_latent_loss + self.commitment_cost * e_latent_loss
    #     quantized = inputs + (quantized - inputs).detach()
    #     return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices
    
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

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, context):
        # x: [batch_size, seq_len, hidden_size]
        # context: [batch_size, num_docs, hidden_size]
        context = context.unsqueeze(1)
        batch_size, seq_len, hidden_size = x.shape
        context = context.view(batch_size, -1, hidden_size)  # Flatten num_docs
        attn_output, _ = self.attention(x, context, context)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
    
# SEDD_BERT: Main model class combining BERT with SEDD components
class SEDD_BERT_rag(nn.Module, PyTorchModelHubMixin):
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
        
        # Timestamp embedder
        self.timestamp_embed = TimestepEmbedder(bert_config.hidden_size)

        # Output layer
        self.output_layer = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
        self.output_layer.weight.data.zero_()
        self.output_layer.bias.data.zero_()

        self.use_style_embedding = config['model'].get('use_style_embedding', True)
        # Add vector quantizer for style codebook
        if self.use_style_embedding:
            self.style_quantizer = VectorQuantizer(num_embeddings=2, embedding_dim=bert_config.hidden_size)
            self.style_proj = nn.Linear(50258, 2048).bfloat16() 
            self.style_proj2 = nn.Linear(2048, 50258)
            self.fusion = PerFuMe(1024)
        else:
            self.style_quantizer = None
            
        self.cross_attention = CrossAttentionLayer(bert_config.hidden_size)
            
    # Forward pass of the SEDD_BERT model
    def forward(self, input_ids, sigma, style_codebook=None, context = None):
        # Embed timestamps
        timestamp_embeddings = self.timestamp_embed(sigma)

        # BERT forward pass
        outputs = self.bert(input_ids, output_hidden_states=True)
        sequence_output = outputs.last_hidden_state
        print("sequence output", sequence_output.shape)
        
        
        # print("context", context)
        # Add retrieved context via cross-attention
        if context is not None:
            # print("context ", context.shape)
            sequence_output = self.cross_attention(sequence_output, context)
        # Add timestamp embeddings to BERT output
        sequence_output += timestamp_embeddings.unsqueeze(1)

        # if self.use_style_embedding:
        #     # get codebook
        #     style_codebook = self.style_quantizer()
            
            
        #     # Fuse quantized style with sequence output
        #     # sequence_output += style_codebook.unsqueeze(1)
        #     # sequence_output = self.fusion(sequence_output, style_codebook)
        #     x = self.style_proj(sequence_output)
        #     # print("after transformation x: ", x.shape)
        #     # fusion of input and codebook
        #     x = self.fusion(x, style_codebook)
        #     # print("after fusion:", x.shape)
        #     # converting back to vocab size dimension
        #     sequence_output = self.style_proj2(x)
        
        
            
        # print("After adding sequence output:  ", sequence_output.shape)
        # Final output layer
        logits = self.output_layer(sequence_output)

        # Apply sigma correction for diffusion process
        esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(logits.dtype)[:, None, None]
        logits = logits - esigm1_log - torch.log(torch.tensor(logits.shape[-1] - 1, dtype=logits.dtype, device=logits.device))

        # Zero out logits for absorbing state
        logits = torch.scatter(logits, -1, input_ids.unsqueeze(-1), torch.zeros_like(logits[..., :1]))

        return logits

# Score function: Computes the score for the diffusion process
def score_fn(model, x, sigma, style_codebook, context, train=True, sampling=False):
    sigma = sigma.reshape(-1)
    if train:
        model.train()
    else:
        model.eval()
    # print("cony", context)
    score = model(x, sigma, style_codebook, context)

    if sampling:
        return score.exp()
    return score
