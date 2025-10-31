import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BertModel, BertTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
# from generate import generate
from itertools import chain
from generate_vq import generate

# --- Component classes from diffusion_train.py ---
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs, style_labels=None):
        # Assuming inputs shape is [batch_size, embedding_dim]
        if inputs.dim() > 2:
             inputs = inputs.squeeze(-1).squeeze(-1)
        
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
    """PerStyM fusion module."""
    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.transform_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*hidden_size, 2*hidden_size),
                nn.GELU(),
                nn.LayerNorm(2*hidden_size)
            ) for _ in range(num_layers)
        ])
        self.semantic_gate = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.Sigmoid())
        self.style_gate = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.Sigmoid())

    def forward(self, zs, ef):
        ef = ef.unsqueeze(1).expand_as(zs)
        combined = torch.cat([zs, ef], dim=-1)
        for layer in self.transform_layers:
            transformed = layer(combined)
            combined = combined + transformed
        μ_s = self.semantic_gate(combined)
        μ_i = self.style_gate(combined)
        z_sem = (1 - μ_s) * zs + μ_s * combined[..., :zs.size(-1)]
        z_int = (1 - μ_i) * zs + μ_i * ef
        fused_output = z_sem + z_int
        return fused_output

class SccgInferenceModel(nn.Module):
    """Encapsulates the full SCCG model for clean inference."""
    def __init__(self, llada_model, style_encoder, style_quantizer, style_projection, fusion_module, output_layer, pre_quant_projection = None):
        super().__init__()
        self.llada_model = llada_model
        self.style_encoder = style_encoder
        self.style_quantizer = style_quantizer
        self.style_projection = style_projection
        self.fusion = fusion_module
        self.output_layer = output_layer
        self.pre_quant_projection = pre_quant_projection

    def forward(self, llada_input_ids, bert_input_ids, style_label =None):
        # 1. Get hidden states from LLaDA
        llada_outputs = self.llada_model(input_ids=llada_input_ids, output_hidden_states=True)
        sequence_output = llada_outputs.hidden_states[-1]

        # 2. Get style embedding from BERT
        style_encoder_outputs = self.style_encoder(input_ids=bert_input_ids)
        style_embedding_mean = torch.mean(style_encoder_outputs.last_hidden_state, dim=1)

        if self.pre_quant_projection is not None:
            style_embedding_mean = self.pre_quant_projection(style_embedding_mean)
        
        # 3. Quantize style embedding
        _, quantized_style_vec, _ = self.style_quantizer(style_embedding_mean, style_label)

        # 4. Project style vector
        style_codebook = self.style_projection(quantized_style_vec)

        # 5. Fuse representations
        fused_hidden_states = self.fusion(sequence_output, style_codebook)
        
        # 6. Get final logits
        logits = self.output_layer(fused_hidden_states)

        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return ModelOutput(logits=logits)

def set_cuda_device(gpu_id):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("CUDA is not available. Using CPU.")

def load_sccg_model_and_tokenizers(base_model_name, sslcl_checkpoint_path, sccg_checkpoint_path, device, latent_code_vector_dim):
    """Load the base model and all components for SCCG inference."""
    print(f"Loading base LLaDA model: {base_model_name}")
    
    # --- Load LLaDA model and tokenizer ---
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    
    base_llada_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map={"": device}, trust_remote_code=True, quantization_config=bnb_config, torch_dtype=torch.bfloat16)
    llada_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    special_tokens = {
            "pad_token": "[PAD]", # Explicitly set pad token
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "additional_special_tokens": ["<start_id>", "<end_id>", "<eot_id>", "[MASK]"]
        }
    llada_tokenizer.add_special_tokens(special_tokens)
    base_llada_model.resize_token_embeddings(len(llada_tokenizer))
    base_llada_model.config.pad_token_id = llada_tokenizer.pad_token_id
    llada_model = get_peft_model(base_llada_model, lora_config)

    # --- Load Style Encoder (BERT) ---
    print("Loading Style Encoder (BERT)...")
    bert_model_name = 'bert-base-uncased'
    style_encoder = BertModel.from_pretrained(bert_model_name).to(device)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # --- Instantiate additional modules ---
    llada_hidden_size = llada_model.config.hidden_size
    bert_hidden_size = style_encoder.config.hidden_size
    
    # If latent_code_vector_dim != 768, add a projection before quantizer
    if latent_code_vector_dim != bert_hidden_size:
        pre_quant_projection = nn.Linear(bert_hidden_size, latent_code_vector_dim).to(device)
    else:
        pre_quant_projection = None
        
    style_quantizer = VectorQuantizer(num_embeddings=2, embedding_dim=latent_code_vector_dim).to(device)
    style_projection = nn.Linear(latent_code_vector_dim, llada_hidden_size).to(device)
    fusion_module = PerStyM(llada_hidden_size).to(device)
    output_layer = nn.Linear(llada_hidden_size, llada_model.config.vocab_size).to(device)

    # --- Load state dicts from checkpoints ---
    print(f"Loading SSLCL checkpoint: {sslcl_checkpoint_path}")
    sslcl_checkpoint = torch.load(sslcl_checkpoint_path, map_location=device)
    style_quantizer.load_state_dict(sslcl_checkpoint['style_quantizer_state_dict'])

    print(f"Loading SCCG checkpoint: {sccg_checkpoint_path}")
    sccg_checkpoint = torch.load(sccg_checkpoint_path, map_location=device)
    
    llada_model.load_state_dict(sccg_checkpoint['model_state_dict'], strict=False)
    style_projection.load_state_dict(sccg_checkpoint['style_projection_state_dict'])
    fusion_module.load_state_dict(sccg_checkpoint['fusion_state_dict'])
    output_layer.load_state_dict(sccg_checkpoint['output_layer_state_dict'])

    # --- Assemble final model ---
    model = SccgInferenceModel(llada_model, style_encoder, style_quantizer, style_projection, fusion_module, output_layer, pre_quant_projection).to(device).eval()

    print("Model loading complete.")
    return model, llada_tokenizer, bert_tokenizer

def get_zero_shot_prompt(hs, style):
    return f"Task: Generate counterspeech for the hatespeech.\n\nHatespeech:\n{hs}\n\nStyle: {style}\n\nCounterspeech:"

def generate_model_response(model, llada_tokenizer, bert_tokenizer, prompt, gen_length, steps, block_length, style_label):
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = llada_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    llada_input_ids = llada_tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    bert_input_ids = bert_tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True, padding="max_length").input_ids.to(device)

    print("Generating response...")
    output = generate(
        model,
        llada_input_ids,
        bert_input_ids=bert_input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.6,
        cfg_scale=0.0,
        remasking='low_confidence',
        tokenizer=llada_tokenizer,
        style_label=style_label
    )

    response = llada_tokenizer.batch_decode(output[:, llada_input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

def main(args):
    set_cuda_device(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, llada_tokenizer, bert_tokenizer = load_sccg_model_and_tokenizers(
        args.model_name,
        args.sslcl_checkpoint_path,
        args.sccg_checkpoint_path,
        device,
        args.latent_code_vector_dim
    )

    input_data = pd.read_csv(args.input_file)
    # input_data = input_data[:10]
    print(input_data['Hatespeech'].head())
    input_data = input_data
    results = []
    print("\nStarting inference...")

    with torch.no_grad():
        for _, row in tqdm(input_data.iterrows(), total=len(input_data)):
            prompt = get_zero_shot_prompt(row["Hatespeech"], row["Style"])
            if row["Style"] == "Gandhi":
                style_label_int = 0
            else:
                style_label_int = 1
            
            # Convert label to a tensor on the correct device
            style_label_tensor = torch.tensor([style_label_int], dtype=torch.long, device=device)
            
            response = generate_model_response(model, llada_tokenizer, bert_tokenizer, prompt, args.gen_length, args.steps, args.block_length, style_label=style_label_tensor)
            results.append({"hatespeech": row["Hatespeech"], "style": row['Style'], "output": row.get("Counterspeech", "N/A"), "CS": response})
            print(f"\nInput: {row['Hatespeech']}\nGenerated Response: {response}\n" + "-" * 80)

    output_dir = f"results/sccg_sft/{args.latent_code_vector_dim}/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.filename}.csv")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nInference completed. Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned SCCG model")
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--latent_code_vector_dim", type=int, default=768, help="Codebook embedding dimension (must match training)")
    parser.add_argument("--sslcl_checkpoint_path", type=str, required=True, help="Path to the trained SSLCL checkpoint (.pth)")
    parser.add_argument("--sccg_checkpoint_path", type=str, required=True, help="Path to the trained SCCG checkpoint (.pth)")
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    main(args)

