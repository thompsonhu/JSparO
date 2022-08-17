## JSparO
Joint sparse optimization (JSparO) has been introduced to solve some application problems with multiple measurement signals. This R packages aims to solve the joint sparse optimization with different penalties $\ell_{p,q}$ norm:

$$\min_{X \in \mathbb{R}^{n\times\ t}}\ ||AX - B||^2_F + \lambda ||X||^{q}_{p,q},$$

via proximal gradient methods. The designed approach also has been applied in cell fate conversion prediction.



### Usage of JSparO

### For Matlab

1. First download the folder `Matlab` in PC.

2. Run the `main.m` file in folder `Matlab` to get all solutions of joint sparse optimization with different regularizers $\ell_{p,q}$ norm.

3. Find the solutions in created folder `Outputs`.



### For R

1. One may install the **JSparO** R package from Github by running command in R:

``` r
install.packages("devtools")
devtools::install_github("thompsonhu/JSparO/R")
```

2. Once the R pacakage is installed, please refer to the reference manual, e.g. inside R console type:

```r
library(JSparO)
?demo_JSparO
```



### Example: solve JSparO problem with $\ell_{2,1/2}$ norm

``` r
library(JSparO)
m <- 256; n <- 1024; t <- 100
A0 <- matrix(rnorm(m * n), nrow = m, ncol = n)
B0 <- matrix(rnorm(m * t), nrow = m, ncol = t)
X0 <- matrix(0, nrow = n, ncol = t)
res_JSparO <- demo_JSparO(A0, B0, X0, s = 10, p = 2, q = 'half', maxIter = 50)
```