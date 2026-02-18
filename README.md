# ConformalSPC
Repository for the paper: Distribution-Free Process Monitoring with Conformal Prediction (https://arxiv.org/abs/2512.23602)

The files q1, q2, and q3 are the raw Python used to generate the visuals within the paper itself.

Provided in addition is a small example library (conformalSPC.py) with an example file (example.py) that demonstrates some of the concepts within the paper. Do note the the code is ment for demonstration only, and is optimized or exhaustively bug-tested and should not be used in production without further testing/adjustments. In general, the implementation here should have its core components replaced using existing optimized and tested tools (the MAPIE, StatsForecast, and TorchCP libraries are some options).

For those unfamilar with conformal prediction, the paper _A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification_ (https://arxiv.org/abs/2107.07511) is highly recommended for a first (fairly rigorous) exposure to the ideas. 

