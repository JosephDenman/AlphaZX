import torch
from torch.distributions import Distribution, Categorical, Independent, Bernoulli, MixtureSameFamily

from models.utils import rand_transfer_edge_probs

params = torch.tensor(
    [[
        [[0.1477, 0.3763, 0.0892, 0.8747, 0.5743],
         [0.3361, 0.7498, 0.6120, 0.3713, 0.1393],
         [0.2110, 0.6862, 0.9630, 0.3793, 0.4347],
         [0.3052, 0.3073, 0.5056, 0.2214, 0.7221]],

        [[0.1716, 0.4645, 0.6109, 0.0000, 0.0000],
         [0.8284, 0.0114, 0.3205, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]],

        [[[0.5540, 0.6505, 0.2352, 0.0000, 0.0000],
          [0.4460, 0.8075, 0.0269, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],

         [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]
    ])


class MultivariateBernoulliMixture(Distribution):
    def __init__(self, params: torch.Tensor):
        """
        A multivariate Bernoulli mixture distribution.
        :param params: B x N x E_incident x (E_incident + 1) tensor of Bernoulli mixture parameters. Each first entry in
                       the second to last dimension is a mixture parameter, and each entry after the first entry in the
                       final dimension is a parameter to a Bernoulli distribution, i.e., a multivariate Bernoulli distribution.
                       The mixture parameters and Bernoulli parameters are combined to form a mixture of Bernoulli distributions of
                       size |E_incident| with events that are |E_incident| long 1D binary tensors.
        """
        super().__init__(torch.Size(params.shape[:2]), torch.Size([params.shape[-2]]), validate_args=False)
        # noinspection PyTypeChecker
        if not torch.all((params >= 0.0) & (params <= 1.0)):
            raise ValueError('All parameters must be in the range [0, 1].')
        # Mixture coefficients for each mixture component
        mixture_coeffs = params[..., 0]
        if not torch.allclose(mixture_coeffs.sum(dim=-1), torch.ones_like(mixture_coeffs[..., 0])):
            raise ValueError('Mixture parameters must sum to 1.')
        # Bernoulli parameters for each component
        bernoulli_params = params[..., 1:]
        # Create categorical distribution for selecting the mixture components
        mixture_dist = Categorical(probs=mixture_coeffs)
        # Create the Bernoulli distribution for each mixture component
        # Since each component is multivariate Bernoulli, we treat each independently
        component_dist = Independent(Bernoulli(probs=bernoulli_params), 1)
        # Combine into a mixture distribution
        self.params = params
        self.dist = MixtureSameFamily(mixture_dist, component_dist)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        samples = self.dist.sample(sample_shape)
        if samples.shape[-1] != self.params.shape[-1] - 1:
            raise ValueError(f'The last dimension of the sample {samples} must be one less than the last dimension of the parameters.')
        return samples


for b in range(1, 5):
    for n in range(1, 5):
        for e in range(1, 5):
            params = rand_transfer_edge_probs(b, n, e)
            print('params = ', params)
            dist = MultivariateBernoulliMixture(params)
            sample = dist.sample()
            print('sample = ', sample)
            prob = dist.log_prob(sample).exp()
            print('prob = ', prob)