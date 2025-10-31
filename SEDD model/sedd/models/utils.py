import torch
import torch.nn.functional as F


def get_model_fn(model, train=False, style_codebook = None, style = None, target=None):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability

    Returns:
        A model function.
    """

    def model_fn(x, sigma, style_codebook, style, target):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()

            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma, style_codebook, style, target)

    return model_fn


def get_score_fn(model, train=False, sampling=False, style_codebook=None, style=None, target = None):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train, style_codebook= style_codebook, style = style, target = target)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(x, sigma, style_codebook, style, target):
            sigma = sigma.reshape(-1)
            score, vq_loss = model_fn(x, sigma, style_codebook,style, target )

            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp(), vq_loss

            return score, vq_loss

    return score_fn