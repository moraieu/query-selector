# Query Selector
Query Selector is a sparse attention layer proposed in https://arxiv.org/pdf/2107.08687v1.pdf

# Depencency
```
Python            3.7.9
deepspeed         0.4.0
numpy             1.20.3
pandas            1.2.4
scipy             1.6.3
tensorboardX      1.8
torch             1.7.1
torchaudio        0.7.2
torchvision       0.8.2
tqdm              4.61.0
```

# Results on ETT dataset
## Univariate
| Data | Prediction len | Informer MSE | Informer MAE | Trans former MSE | Trans former MAE | Query Selector MSE | Query Selector MAE |  MSE ratio |
| --- | ---  |  --- | --- | --- | --- | --- | --- | --- | 
| ETTh1 |   24 | 0.0980 | 0.2470 | 0.0548 | 0.1830 |  **0.0436** | **0.1616** | **0.445** |
| ETTh1 |   48 | 0.1580 | 0.3190 | 0.0740 | 0.2144 |  **0.0721** | **0.2118** | **0.456** |
| ETTh1 |  168 | 0.1830 | 0.3460 | 0.1049 | 0.2539 |  **0.0935** | **0.2371** | **0.511** |
| ETTh1 |  336 | 0.2220 | 0.3870 | 0.1541 | 0.3201 |  **0.1267** | **0.2844** | **0.571** |
| ETTh1 |  720 | 0.2690 | 0.4350 | 0.2501 | 0.4213 |  **0.2136** | **0.3730** | **0.794** |
| ETTh2 |   24 | 0.0930 | 0.2400 | 0.0999 | 0.2479 |  **0.0843** | **0.2239** | **0.906** |
| ETTh2 |   48 | 0.1550 | 0.3140 | 0.1218 | 0.2763 |  **0.1117** | **0.2622** | **0.721** |
| ETTh2 |  168 | 0.2320 | 0.3890 | 0.1974 | 0.3547 |  **0.1753** | **0.3322** | **0.756** |
| ETTh2 |  336 | 0.2630 | 0.4170 | 0.2191 | 0.3805 |  **0.2088** | **0.3710** | **0.794** |
| ETTh2 |  720 | 0.2770 | 0.4310 | 0.2853 | 0.4340 |  **0.2585** | **0.4130** | **0.933** |
| ETTm1 |   24 | 0.0300 | 0.1370 | 0.0143 | 0.0894 |  **0.0139** | **0.0870** | **0.463** |
| ETTm1 |   48 | 0.0690 | 0.2030 | **0.0328** | **0.1388** |  0.0342 | 0.1408 | **0.475** |
| ETTm1 |   96 | 0.1940 | **0.2030** | **0.0695** | 0.2085 |  0.0702 | 0.2100 | **0.358** |
| ETTm1 |  288 | 0.4010 | 0.5540 | **0.1316** | **0.2948** |  0.1548 | 0.3240 | **0.328** |
| ETTm1 |  672 | 0.5120 | 0.6440 | **0.1728** | 0.3437 |  0.1735 | **0.3427** | **0.338** |

## Multivariate
| Data | Prediction len | Informer MSE | Informer MAE | Trans former MSE | Trans former MAE | Query Selector MSE | Query Selector MAE |  MSE ratio |
| --- | ---  |  --- | --- | --- | --- | --- | --- | --- | 
| ETTh1 |   24 | 0.5770 | 0.5490 | 0.4496 | 0.4788 |  **0.4226** | **0.4627** | **0.732** |
| ETTh1 |   48 | 0.6850 | 0.6250 | 0.4668 | 0.4968 |  **0.4581** | **0.4878** | **0.669** |
| ETTh1 |  168 | 0.9310 | 0.7520 | 0.7146 | 0.6325 |  **0.6835** | **0.6088** | **0.734** |
| ETTh1 |  336 | 1.1280 | 0.8730 | **0.8321** | 0.7041 |  0.8503 | **0.7039** | **0.738** |
| ETTh1 |  720 | 1.2150 | 0.8960 | **1.1080** | **0.8399** |  1.1150 | 0.8428 | **0.912** |
| ETTh2 |   24 | 0.7200 | 0.6650 | 0.4237 | 0.5013 |  **0.4124** | **0.4864** | **0.573** |
| ETTh2 |   48 | 1.4570 | 1.0010 | 1.5220 | 0.9488 |  **1.4074** | **0.9317** | **0.966** |
| ETTh2 |  168 | 3.4890 | 1.5150 | **1.6225** | **0.9726** |  1.7385 | 1.0125 | **0.465** |
| ETTh2 |  336 | 2.7230 | 1.3400 | 2.6617 | 1.2189 |  **2.3168** | **1.1859** | **0.851** |
| ETTh2 |  720 | 3.4670 | 1.4730 | 3.1805 | 1.3668 |  **3.0664** | **1.3084** | **0.884** |
| ETTm1 |   24 | 0.3230 | **0.3690** | **0.3150** | 0.3886 |  0.3351 | 0.3875 | **0.975** |
| ETTm1 |   48 | 0.4940 | 0.5030 | **0.4454** | **0.4620** |  0.4726 | 0.4702 | **0.902** |
| ETTm1 |   96 | 0.6780 | 0.6140 | 0.4641 | **0.4823** |  **0.4543** | 0.4831 | **0.670** |
| ETTm1 |  288 | 1.0560 | 0.7860 | 0.6814 | 0.6312 |  **0.6185** | **0.5991** | **0.586** |
| ETTm1 |  672 | 1.1920 | 0.9260 | 1.1365 | 0.8572 |  **1.1273** | **0.8412** | **0.946** |

## State Of Art

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth1-24)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-24?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth1-48)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-48?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth1-168)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-168?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth1-336)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-336?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth1-720)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720?p=long-term-series-forecasting-with-query)


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth2-24)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-24?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth2-48)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-48?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth2-168)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-168?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth2-336)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-336?p=long-term-series-forecasting-with-query)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/long-term-series-forecasting-with-query/time-series-forecasting-on-etth2-720)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-720?p=long-term-series-forecasting-with-query)

# Citation
```
@misc{klimek2021longterm,
      title={Long-term series forecasting with Query Selector -- efficient model of sparse attention}, 
      author={Jacek Klimek and Jakub Klimek and Witold Kraskiewicz and Mateusz Topolewski},
      year={2021},
      eprint={2107.08687},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
