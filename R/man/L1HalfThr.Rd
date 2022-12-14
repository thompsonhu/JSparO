% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/L1HalfThr.R
\name{L1HalfThr}
\alias{L1HalfThr}
\title{L1HalfThr - Iterative Half Thresholding Algorithm based on \eqn{l_{1,1/2}} norm}
\usage{
L1HalfThr(A, B, X, s, maxIter = 200)
}
\arguments{
\item{A}{Gene expression data of transcriptome factors (i.e. feature matrix in machine learning).
The dimension of A is m * n.}

\item{B}{Gene expression data of target genes (i.e. observation matrix in machine learning).
The dimension of B is m * t.}

\item{X}{Gene expression data of Chromatin immunoprecipitation or other matrix
(i.e. initial iterative point in machine learning). The dimension of X is n * t.}

\item{s}{joint sparsity level}

\item{maxIter}{maximum iteration}
}
\value{
The solution of proximal gradient method with \eqn{l_{1,1/2}} regularizer.
}
\description{
The function aims to solve \eqn{l_{1,1/2}} regularized least squares.
}
\details{
The L1HalfThr function aims to solve the problem:
\deqn{\min \|AX-B\|_F^2 + \lambda \|X\|_{1,1/2}}
to obtain s-joint sparse solution.
}
\examples{
m <- 256; n <- 1024; t <- 5; maxIter0 <- 50
A0 <- matrix(rnorm(m * n), nrow = m, ncol = n)
B0 <- matrix(rnorm(m * t), nrow = m, ncol = t)
X0 <- matrix(0, nrow = n, ncol = t)
NoA <- norm(A0, '2'); A0 <- A0/NoA; B0 <- B0/NoA
res_L1half <- L1HalfThr(A0, B0, X0, s = 10, maxIter = maxIter0)

}
\author{
Xinlin Hu \href{mailto:thompson-xinlin.hu@connect.polyu.hk}{thompson-xinlin.hu@connect.polyu.hk}

Yaohua Hu \href{mailto:mayhhu@szu.edu.cn}{mayhhu@szu.edu.cn}
}
