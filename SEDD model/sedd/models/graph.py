
import abc
import torch
import torch.nn.functional as F
from sedd.models.catsample import sample_categorical


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)

class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """

        # print(f"i from the reverse rate {i}")
        # print(f"transp_rate(i) from the reverse rate {self.transp_rate(i)}")
        # print(f"score from the reverse rate {score.shape}")

        normalized_rate = self.transp_rate(i) * score

        # print(f"normalized_rate from the reverse rate {normalized_rate.shape}")
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        # print(f"normalized_rate : {normalized_rate}")
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)


    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass


    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class AbsorbingGraph(Graph):
    def __init__(self, vocab_size):
        super().__init__()
        self._dim = vocab_size + 1 # for the absorbing state

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        
        edge[i == self.dim - 1] += 1

        #print(f"Dim of edge : {edge}")
        return edge

    def transition(self, i, sigma):
        pass

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        # Compute probability of moving to the absorbing state
        move_chance = 1 - (-sigma).exp()
        # q will contain list of random probabilities where the list is of size number of tokens
        q = torch.rand(*i.shape, device=i.device)
        # Here only those tokens will get converted or flipped which has probabilities less then the move chance
        move_indices = q < move_chance
        # print(f"Generated random probabilities : {q}")
        # print("move_indices", move_indices)
        # Here it will simply perturb the 50257 token on the selected tokens
        i_pert = torch.where(move_indices, self.dim - 1, i)
        #print("i_pert", i_pert)
        return i_pert

    # def sample_transition(self, i, sigma):
    #     """
    #     Samples a state from the transition distribution for state i at time sigma.
    #     Args:
    #         i: Input states, shape [...]
    #         sigma: Time parameter, shape [...]
    #         tau: Temperature parameter controlling randomness (default: 1.0)
    #     Returns:
    #         Sampled states, shape [...]
    #     """
    #     # Compute probability of moving to the absorbing state
    #     move_chance = 1 - (-sigma).exp()  # Shape: [...]
        
    #     # Create logits for staying vs. moving
    #     logits = torch.stack(
    #         [torch.log(1 - move_chance + 1e-10), torch.log(move_chance + 1e-10)],
    #         dim=-1
    #     )  # Shape: [..., 2]
    #     tau = 0.5
    #     # Apply temperature scaling
    #     logits = logits / tau
        
    #     # Sample from the categorical distribution
    #     move_indices = torch.distributions.Categorical(logits=logits).sample()  # Shape: [...]
        
    #     # If move_indices == 1, transition to absorbing state; otherwise, stay
    #     i_pert = torch.where(move_indices == 1, self.dim - 1, i)
        
    #     return i_pert

    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    # x is perturbed, x0 is the original
    def score_entropy(self, score, sigma, x, x0):
        # This is where the sample has been absorbed / perturbed
        # [[ True, False,  True,  True,  True, False, False,  True]]
        rel_ind = x == self.dim - 1
        # print("rel_ind", rel_ind.shape, rel_ind)
        
        #print(f"rel_ind : {rel_ind}")
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        #print("esigm1", esigm1.shape, esigm1)

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        # print("ratio", ratio.shape, ratio)
        
        other_ind = x0[rel_ind]
        # print("other_ind", other_ind.shape, other_ind)

        # print(f"This is from score_entropy from Graph.py, other_ind : {other_ind}")
        # print(f"This is from score_entropy from Graph.py, perturbed esigm1.expand_as(x)[rel_ind] : {esigm1.expand_as(x)[rel_ind]}")
        # print(f"This is from score_entropy from Graph.py, perturbed ratio : {ratio}")


        # print(f"score[rel_ind] : {score[rel_ind]}")
        # print(f"other_ind[..., None] : {other_ind[..., None]}")
        # # negative_term

        # print("Values of torch.gather")
        # print(torch.gather(score[rel_ind], -1, other_ind[..., None]))
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
        # print("neg_term", neg_term.shape, neg_term)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        # print("pos_term", pos_term.shape, pos_term)

        # constant term
        const = ratio * (ratio.log() - 1)
        # print("const", const.shape, const)

        entropy = torch.zeros(*x.shape, device=x.device) 
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy

# P(x_t|x_0) = Is the peturbation distribution related to Q graph
# P(y|x_0) = is the transition distribution related to score_transition.