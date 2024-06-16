from ding.model import IModelWrapper

from alphazx.distributions import AlphaZXDistribution


class AlphaZXDistributionWrapper(IModelWrapper):
    def forward(self, *args, **kwargs):
        output = self._model.forward(*args, **kwargs)
        assert isinstance(output, dict), "model output must be dict, but found {}".format(type(output))
        logit = output['logit']  # logit: AlphaZXDistributionParams
        azx_dist = AlphaZXDistribution(logit)
        return {'action': azx_dist.sample(1)}
