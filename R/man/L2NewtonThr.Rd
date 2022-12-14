% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/L2NewtonThr.R
\name{L2NewtonThr}
\alias{L2NewtonThr}
\title{L2NewtonThr - Iterative Thresholding Algorithm based on \eqn{l_{2,q}} norm with Newton method}
\usage{
L2NewtonThr(A, B, X, s, q, maxIter = 200, innMaxIter = 30, innEps = 1e-06)
}
\arguments{
\item{A}{Gene expression data of transcriptome factors (i.e. feature matrix in machine learning).
The dimension of A is m * n.}

\item{B}{Gene expression data of target genes (i.e. observation matrix in machine learning).
The dimension of B is m * t.}

\item{X}{Gene expression data of Chromatin immunoprecipitation or other matrix
(i.e. initial iterative point in machine learning). The dimension of X is n * t.}

\item{s}{joint sparsity level}

\item{q}{value for \eqn{l_{2,q}} norm (i.e., 0 < q < 1)}

\item{maxIter}{maximum iteration}

\item{innMaxIter}{maximum iteration in Newton step}

\item{innEps}{criterion to stop inner iteration}
}
\value{
The solution of proximal gradient method with \eqn{l_{2,q}} regularizer.
}
\description{
The function aims to solve \eqn{l_{2,q}} regularized least squares, where the proximal optimization subproblems will be solved by Newton method.
}
\details{
The L2NewtonThr function aims to solve the problem:
\deqn{\min \|AX-B\|_F^2 + \lambda \|X\|_{2,q}}
to obtain s-joint sparse solution.
}
\examples{
m <- 256; n <- 1024; t <- 5; maxIter0 <- 50
A0 <- matrix(rnorm(m * n), nrow = m, ncol = n)
B0 <- matrix(rnorm(m * t), nrow = m, ncol = t)
X0 <- matrix(0, nrow = n, ncol = t)
NoA <- norm(A0, '2'); A0 <- A0/NoA; B0 <- B0/NoA
res_L2q <- L2NewtonThr(A0, B0, X0, s = 10, q = 0.2, maxIter = maxIter0)

}
\author{
Xinlin Hu \href{mailto:thompson-xinlin.hu@connect.polyu.hk}{thompson-xinlin.hu@connect.polyu.hk}

Yaohua Hu \href{mailto:mayhhu@szu.edu.cn}{mayhhu@szu.edu.cn}
}
