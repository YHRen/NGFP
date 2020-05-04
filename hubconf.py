# Optional list of dependencies required by the package
dependencies = ['torch']

from NeuralGraph.model import QSAR
from torch.utils.model_zoo import load_url

model_urls={'Mpro':
            "https://github.com/YHRen/NGFP/blob/master/pretrained/MPro_mergedmulti_class.pkg",
            "6vww":
            "https://github.com/YHRen/NGFP/blob/master/pretrained/6vww_sample20kmulti_class.pkg"}


supported_protrains = set(model_urls.keys())

def nfp(pretrained=False, protein="Mpro", progress=True, **kwargs):
    r"""Pretrained NFP model
    """
    assert protein in supported_protrains
    if pretrained:
        model = load_url(model_urls[protein], progress=progress)
    else:
        model = QSAR(**kwargs)
    return model
