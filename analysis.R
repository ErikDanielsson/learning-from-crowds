
set.seed(136)

rmixnorm <- function(N, m1, m2, s1, s2, t){
  ind <- I(runif(N) > t)
  X <- rep(0,N)
  X[ind] <- rnorm(sum(ind), m1, s1)
  X[!ind] <- rnorm(sum(!ind), m2, s2)
  return(X)
}

dmixnorm <- function(x, m1, m2, s1, s2, t){
  y <- (1-t)*dnorm(x,m1,s1) + t*dnorm(x,m2,s2)
  return(y)
}

em_alg_gmm <- function(x, m1_est, m2_est, s1_est, s2_est, t_est) {
  n = length(x)
  l_old = -10000000
  repeat{
    l_new = sum( log(( 1 - t_est ) * ( dnorm(x, m1_est, s1_est) ) + 
      (t_est) * ( dnorm(x, m2_est, s2_est) ) ) )
    if(l_new - l_old < (1e-15) | is.na(l_new)==TRUE | is.infinite(l_new)==TRUE) {
       break
    }
    p_n = ( t_est * dnorm(x, m2_est, s2_est) ) / dmixnorm(x, m1_est, m2_est, s1_est, s2_est, t_est)
    l_old = l_new
    t_est = (1/n) * ( sum(p_n) )
    m1_est = ( sum( (1-p_n) * x) / sum(1-p_n) )
    s1_est = sqrt(( sum( (1-p_n) * (x - m1_est)^2 ) / sum(1-p_n) ))
    m2_est = ( sum( (p_n) * x) / sum(p_n) )
    s2_est = sqrt(( sum( (p_n) * (x - m2_est)^2 ) / sum(p_n) ))
  }
  return(c(m1_est, m2_est, s1_est, s2_est, t_est))
}

n = 2000
mu1 = 2
mu2 = 5
sigma1 = 5
sigma2 = 3
tau = 0.3

X = rmixnorm(n, mu1, mu2, sigma1, sigma2, tau)

mu1_est = 1
mu2_est = 4
sigma1_est = 100
sigma2_est = 2
tau_est = 0.2

theta = em_alg_gmm(X, mu1_est, mu2_est, sigma1_est, sigma2_est, tau_est)
