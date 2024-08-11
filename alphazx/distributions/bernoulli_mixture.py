import torch
from torch.distributions import Categorical, Independent, Bernoulli, MixtureSameFamily


class MultivariateBernoulli:
    def __init__(self, params: torch.Tensor):
        """
        A multivariate Bernoulli mixture distribution.
        :param params: B x N x E_incident tensor of Bernoulli mixture parameters. Each entry in the last dimension is a
                       parameter to a Bernoulli distribution, i.e., a multivariate Bernoulli distribution.
        """
        self.params = params
        # noinspection PyTypeChecker
        if not torch.all((params >= 0.0) & (params <= 1.0)):
            raise ValueError('All parameters must be in the range [0, 1].')
        # Since each component is multivariate Bernoulli, we treat each independently
        self.dist = Independent(Bernoulli(probs=self.params, validate_args=True), 1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.dist.sample(sample_shape)


class MultivariateBernoulliMixture:
    def __init__(self, params: torch.Tensor):
        """
        A multivariate Bernoulli mixture distribution.
        :param params: B x N x E_incident x (E_incident + 1) tensor of Bernoulli mixture parameters. Each first entry in
                       the second to last dimension is a mixture parameter, and each entry after the first entry in the
                       final dimension is a parameter to a Bernoulli distribution, i.e., a multivariate Bernoulli distribution.
                       The mixture parameters and Bernoulli parameters are combined to form a mixture of Bernoulli distributions of
                       size |E_incident| with events that are |E_incident| long 1D binary tensors.
        """
        self.params = params
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
        self.dist = MixtureSameFamily(mixture_dist, component_dist)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        samples = self.dist.sample(sample_shape)
        if samples.shape[-1] != self.params.shape[-1] - 1:
            raise ValueError(
                f'The last dimension of the sample {samples} must be one less than the last dimension of the parameters.')
        return samples
