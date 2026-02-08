myfunc <- function(v_s, i_v, iter)
{
  v_mat <- matrix(NA, nrow(v_s), ncol(v_s))
  
  for (i in 1:nrow(v_s))
  {
    for (j in 1:ncol(v_s))
    {
      d_val = round(i_v[i]%%256) 
      v_mat[i, j] = v_s[i, j]*(sin(d_val)*sin(d_val)-cos(d_val)*cos(d_val))/cos(iter);
    }
  }
  return(v_mat)
}

N1 <- 1e3; N2 <- 2e3; N_tot <- 64
vi_v <- rep(NA, N1)
vd_s <- matrix(NA, N1, N2)

set.seed(123)
for (i in 1:N1){
  vi_v[i] = i + rnorm(1, sd = sqrt(i)*0.01);
  for (j in 1:N2)
    vd_s[i,j] = j + i
}
Res <- rep(NA, N_tot)

# start benchmark
ptm <- proc.time()

# iterative test loop
for (iter in 1:N_tot)
{
  res_mat <- myfunc(vd_s, vi_v, iter)
  Res[iter] <- mean(res_mat)
}

# end benchmark
proc.time() -ptm