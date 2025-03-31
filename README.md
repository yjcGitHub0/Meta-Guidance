
## Figure 1: Illustration of NSG Construction
![Logo](https://github.com/yjcGitHub0/Meta-Guidance/blob/master/NSG.png "NSG")

In this example, the 4th and 7th points in the time series are missing. Given a hyperparameter setting of $r=3$, each missing points will influence its three nearest neighbors, weighted by a Gaussian function $\psi(i)$. Specifically, if the 4th position is missing, the 3rd position receives a weight of $\psi(1)$, the 2nd position $\psi(2)$, and the 1st position $\psi(3)$, following the same pattern for other missing points. After applying this weighting scheme to all missing points, the resulting sequence constitutes the NSG.

## Figure 2: Illustration of PG Construction
![Logo](https://github.com/yjcGitHub0/Meta-Guidance/blob/master/PG.png "PG")

The construction of PG follows a similar approach to NSG. In this example, the original time series has a period of $T=10$, with missing values at the 40th and 70th positions. In PG, the influence of $\psi(i)$ propagates with a step size of $T$. Specifically, if the 40th position is missing, the 30th position receives a weight of $\psi(1)$, the 20th position $\psi(2)$, and the 10th position $\psi(3)$, following the same pattern for other missing points. After applying this weighting scheme to all missing points, the resulting sequence constitutes the PG.

## Figure 3: Diagram of MG Integration into CSDI
![Logo](https://github.com/yjcGitHub0/Meta-Guidance/blob/master/CSDI+MG.png "CSDI+MG")

This figure is adapted from Figure 6 of the original CSDI paper [1]. The full MG sequence is incorporated into CSDI's Side info as a conditional input to the diffusion model. Further implementation details can be found in Appendix A.3.

[1] Tashiro, Yusuke, et al. "Csdi: Conditional score-based diffusion models for probabilistic time series imputation." Advances in neural information processing systems 34 (2021): 24804-24816.


# Based on Time Series Library (TSlib)

This is an earlier version which has a larger time consumption.

run: bash scripts/imputation/exe.sh
