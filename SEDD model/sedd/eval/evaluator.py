import torch
from tqdm import tqdm

from ..trainer.loss import step_fn
from transformers import BertTokenizer, BertModel
from transformers import BartTokenizer, BartModel

class Evaluator:
    def __init__(self, dataset, run, cfg, device = 'cuda'):
        self.dataset = dataset
        self.run = run
        self.cfg = cfg
        self.device = device
        
    def evaluate(self, state, use):
        step = state['step']
        sum_loss = 0
        print(f"Evaluating model on validation set")
        for batch in tqdm(self.dataset):
            # batch = batch.to(self.device)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = self.evaluate_batch(state, batch, use)
            sum_loss += loss.item()
        avg_loss = sum_loss / len(self.dataset)
        print("step: %d, evaluation_loss: %.5e" % (step, avg_loss))
        self.run.track(avg_loss, name='loss', step=state['step'], context={ "subset":"eval" })
        return avg_loss
    
    def evaluate_batch(self, state, batch, use_style_embedding):
        model = state['model']
        model.eval()
        with torch.no_grad():
            device = 'cuda'
            if use_style_embedding:
                # style_embedding = generate_style_embedding(batch).to(device)
                # print(style_embedding.shape)
                style_labels = batch['style'].to(self.device)
                target_labels = batch['target'].to(self.device)
                # getting closest codebook
                # vq_loss, style_codebook, _ = model.style_quantizer(style_embedding.unsqueeze(2).unsqueeze(3), style_labels)
                # style_codebook = style_codebook.squeeze(3).squeeze(2)
            else:
                style_codebook = None
                style_labels = None
                target_labels = None
                vq_loss = torch.tensor(0.0, device=device)
            
            batch_input_ids = batch['input_ids']
            eval_loss, p_batch = step_fn(self.cfg, state, batch_input_ids,use_style_embedding, train=False, style_labels=style_labels, target_labels=target_labels)
        return eval_loss
    