# machine-learning-for-genomic-selection-in-animals

## Datasets Source
[1]Cleveland, M. A., J. M. Hickey, and S. Forni, 2012 A Common Dataset for Genomic Analysis of Livestock Populations. G3 Genes|Genomes|Genetics 2: 429-435.

[2]An, M., G. Zhou, Y. Li, T. Xiang and Y. Ma et al., 2022 Characterization of genetic fundamentals for piglet mortality at birth in Yorkshire, Landrace, and Duroc sows. Animal Genetics 53: 142-145.

[3]Yang, W., J. Wu, J. Yu, X. Zheng and H. Kang et al., 2021 A genome-wide association study reveals additive and dominance effects on growth and  fatness traits in large white pigs. Anim Genet 52: 749-753.

[4]Waldmann, P., Pfeiffer, C., & Mészáros, G. (2020). Sparse convolutional neural networks for genome-wide prediction. Frontiers in Genetics, 11, 25.

## Genomic selection methods used on pig
|Year|Author|ML based methods|Traditional methods|
|----|------------------|-----------------------|-----------------------|
|2013|Tusell L et al.|RKHS regression,RBFNN,BRNN|GBLUP,BayesR,BayesLASSO|
|2018|Waldmann P et al.|MLP|GBLUP,BayesLASSO|
|2020|Waldmann P et al.|CNN|GBLUP,BayesLASSO|
|2022|Wang X et al.|SVR,KRR,RF,Adaboost.R2|GBLUP,ssBLUP,BayesHE|

## Methods Implemented
GBLUP, LASSO, CNN, LSTM, LGBM, Transformer(+AE,+PCA,+ICA,+t-SNE)




