
set.seed(136)
library(geometry)

generate_data <- function(N,w){
   X_1 = runif(N, 0, 1)
   X_2 = runif(N, 0, 1)
   Y = rep(0,N)
   w_matrix = matrix(rep(w, N), nrow=2)
   Y = as.numeric(I(dot(w_matrix,matrix(c(X_1,X_2), nrow=2, byrow=TRUE)) >= 0)) 
   return(data.frame(X_1,X_2,Y))
}

generate_expert <- function(df, alpha, beta){
  expert_df = data.frame()
  for (i in 1:nrow(df)){
    y_i = c()
    if(df$Y[i]==1){
      for (j in 1:length(alpha)){
        y_j = rbern(1,alpha[j])
        y_i = append(y_i, y_j)
      }
    }
    else{
      for (j in 1:length(beta)){
            y_j = rbern(1,1-beta[j])
            y_i = append(y_i, y_j)
      }
    }
    expert_df = rbind(expert_df, data.frame(y_i))
  }
  return(cbind(df,expert_df))
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

# n = 2000
# mu1 = 2
# mu2 = 5
# sigma1 = 5
# sigma2 = 3
# tau = 0.3

# mu1_est = 1
# mu2_est = 4
# sigma1_est = 100
# sigma2_est = 2
# tau_est = 0.2

# theta = em_alg_gmm(X, mu1_est, mu2_est, sigma1_est, sigma2_est, tau_est)

w = c(-1,1)
data = generate_data(10,w)
df = generate_expert(data,1,0)
