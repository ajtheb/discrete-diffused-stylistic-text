from transformers import T5ForConditionalGeneration, T5Tokenizer

class StyleFusedT5(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load your pretrained components
        self.quantizer = VectorQuantizer(num_embeddings=512, embedding_dim=768) 
        self.fusion = PerFuMe(hidden_size=768)
        
        # Freeze all base model parameters
        for param in self.parameters():
            param.requires_grad_(False)

    def fuse_embeddings(self, encoder_outputs):
        """Style fusion process preserving sequence length"""
        # Quantize with style preservation
        _, quantized, _ = self.quantizer(encoder_outputs.last_hidden_state)
        
        # Get style code (global representation)
        style_code = quantized.mean(dim=1)  # [batch_size, hidden_size]
        
        # Fuse with original encoder outputs
        fused = self.fusion(encoder_outputs.last_hidden_state, style_code)
        return fused

    def generate(self, input_ids, **kwargs):
        # Original encoder processing
        encoder_outputs = self.encoder(input_ids=input_ids)
        
        # Fuse quantized style embeddings
        fused_hidden_states = self.fuse_embeddings(encoder_outputs)
        
        # Decode with fused representations
        return super().generate(
            encoder_outputs=(fused_hidden_states,),
            **kwargs
        )

# Usage
model = StyleFusedT5.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Load your pretrained quantizer weights
model.quantizer.load_state_dict(torch.load("your_quantizer.pt")) 

# Generate with style fusion
input_text = "Translate to French: Hello world"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
