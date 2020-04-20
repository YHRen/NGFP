from .model import QSAR
import torch
from torch.utils.model_zoo import load_url 

model_urls={'Mpro':
            "https://github.com/YHRen/NGFP/blob/master/pretrained/MPro_mergedmulti_class.pkg?raw=true",
            "6vww":
            "https://github.com/YHRen/NGFP/blob/master/pretrained/6vww_sample20kmulti_class.pkg?raw=true"}


def nfp(pretrained=False, protein="Mpro", progress=True, **kwargs):
    r"""Pretrained NFP model
    """
    if pretrained:
        model = load_url(model_urls[protein], progress=progress)
    else:
        model = QSAR(**kwargs)
    return model

# Alternatively, use requests and tempfile
# import requests
# import tempfile
# tmp  = requests.get(model_urls[protein])
# tmp_f = tempfile.NamedTemporaryFile(mode="wb")
# tmp_f.write(tmp.content)
# model = torch.load(tmp_f.name)
# return model
