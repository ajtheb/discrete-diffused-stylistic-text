import torch
import numpy as np


def loss_fn(batch, model, noise, graph, use, model_type,step, train=True, t=None, perturbed_batch=None, style_labels=None, target_labels = None):
    """
    Batch shape: [B, L] int. D given from graph
    """
    # print("target_labels",target_labels)
    # sampling steps, depending on it, the time step can be smaller or larger
    sampling_eps=1e-3

    if t is None:
        t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
        
    sigma, dsigma = noise(t)
    
    if perturbed_batch is None:
        perturbed_batch = graph.sample_transition(batch, sigma[:, None])
    
    if(model_type=='normal'):
        from sedd.models.sedd import score_fn
    elif(model_type=='bert'):
        from sedd.models.sedd_BERT2 import score_fn
    elif(model_type in  ['bert_clime', 'bert_cogent']):
        from sedd.models.sedd_BERT3 import score_fn
    else:
        from sedd.models.sedd_BERT_ia3 import score_fn
    
    log_score,aux_loss = score_fn(model, perturbed_batch, sigma, use, step, train=train, sampling=False, style_labels=style_labels, target_labels=target_labels)
    
    loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
    
    loss = (dsigma[:, None] * loss).sum(dim=-1)

    if use :
        # Combine diffusion loss and VQ loss / target loss
        total_loss = loss + aux_loss
    else:
        total_loss = loss
        
    
    return total_loss.mean(), perturbed_batch

def step_fn(cfg, state, batch, use, train=True, style_labels = None, target_labels = None):
    model = state['model']
    noise = state['noise']
    graph = state['graph']
    warmup = cfg['optim']['warmup']
    accum = cfg['training']['accum']
    lr = cfg['optim']['lr']
    step = state['step']
    # print("step:",step)
    grad_clip = 1.

    optimizer = state['optimizer']

    model_type = cfg['model'].get('model_type', 'normal')
    
    if train:
        loss, p_batch = loss_fn(batch, model, noise, graph, use, model_type, step, train=True, style_labels=style_labels, target_labels=target_labels)
        loss = loss.mean() / accum
        loss.backward()

        state['step'] += 1
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()
    else:
        with torch.no_grad():
            loss, p_batch = loss_fn(batch, model, noise, graph, use, model_type, step, train=False, style_labels=style_labels, target_labels=target_labels)
            loss = loss.mean()


    return loss, p_batch
