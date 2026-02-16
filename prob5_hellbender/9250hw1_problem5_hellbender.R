## =============================================================
## Problem 5: Code Optimization & Parallel Computing
## Run on Hellbender via SLURM
## =============================================================

library(doParallel)
library(foreach)

set.seed(123)

## =============================================================
## PART 1: Original slow code - report elapsed time
## =============================================================

myfunc <- function(v_s, i_v, iter)
{
  v_mat <- matrix(NA, nrow(v_s), ncol(v_s))
  for (i in 1:nrow(v_s))
  {
    for (j in 1:ncol(v_s))
    {
      d_val = round(i_v[i]%%256) 
      v_mat[i, j] = v_s[i, j]*(sin(d_val)*sin(d_val)-cos(d_val)*cos(d_val))/cos(iter)
    }
  }
  return(v_mat)
}

# Setup data (same as slow_code.R)
N1 <- 1e3; N2 <- 2e3; N_tot <- 64
vi_v <- rep(NA, N1)
vd_s <- matrix(NA, N1, N2)
set.seed(123)
for (i in 1:N1){
  vi_v[i] = i + rnorm(1, sd = sqrt(i)*0.01)
  for (j in 1:N2)
    vd_s[i,j] = j + i
}

# Time the original
Res_original <- rep(NA, N_tot)
ptm <- proc.time()
for (iter in 1:N_tot)
{
  res_mat <- myfunc(vd_s, vi_v, iter)
  Res_original[iter] <- mean(res_mat)
}
time_original <- proc.time() - ptm

cat("=== PART 1: Original Code ===\n")
cat("Elapsed time:", time_original["elapsed"], "seconds\n")
cat("Res values:\n")
print(Res_original)

## =============================================================
## PART 2: Optimized serial code
## =============================================================

# Precompute d_val for each row: round(vi_v[i] %% 256)
d_vals <- round(vi_v %% 256)

# Precompute -cos(2*d) for each row using trig identity:
# sin^2(d) - cos^2(d) = -cos(2d)
row_trig <- -cos(2 * d_vals)

# Debug prints
cat("\nDEBUG: any NA in d_vals?", any(is.na(d_vals)), "\n")
cat("DEBUG: any NA in row_trig?", any(is.na(row_trig)), "\n")
cat("DEBUG: range of d_vals:", range(d_vals), "\n")
cat("DEBUG: first 5 row_trig:", row_trig[1:5], "\n")

# Time the optimized version
Res_opt <- rep(NA, N_tot)
ptm <- proc.time()
for (iter in 1:N_tot)
{
  # row_trig[i] / cos(iter) is the multiplier for row i, same for all columns
  multiplier <- row_trig / cos(iter)    # length N1 vector
  
  # Multiply each row of vd_s by its multiplier
  # sweep applies the vector along rows (MARGIN=1)
  res_mat <- sweep(vd_s, 1, multiplier, FUN = "*")
  
  Res_opt[iter] <- mean(res_mat)
}
time_opt <- proc.time() - ptm

cat("\n=== PART 2: Optimized Serial Code ===\n")
cat("Elapsed time:", time_opt["elapsed"], "seconds\n")
cat("Speedup over original:", round(time_original["elapsed"] / time_opt["elapsed"], 1), "x\n")
cat("Res values:\n")
print(Res_opt)
cat("Max difference from original:", max(abs(Res_opt - Res_original)), "\n")
cat("Results match:", all.equal(Res_opt, Res_original), "\n")

## =============================================================
## PART 3: Parallel code using foreach
## =============================================================

n_cores_grid <- c(4, 8, 16, 32, 64)
par_times <- numeric(length(n_cores_grid))

for (k in seq_along(n_cores_grid)) {
  nc <- n_cores_grid[k]
  
  cl <- makeCluster(nc)
  registerDoParallel(cl)
  
  ptm <- proc.time()
  
  Res_par <- foreach(iter = 1:N_tot, .combine = c) %dopar% {
    multiplier <- row_trig / cos(iter)
    res_mat <- sweep(vd_s, 1, multiplier, FUN = "*")
    mean(res_mat)
  }
  
  par_times[k] <- (proc.time() - ptm)["elapsed"]
  
  stopCluster(cl)
  
  cat("\nCores:", nc, "| Elapsed:", round(par_times[k], 4), "seconds")
}

# Verify parallel results
cat("\n\n=== PART 3: Parallel Results ===\n")
cat("Max difference from original:", max(abs(Res_par - Res_original)), "\n")

cat("\nSummary of parallel times:\n")
for (k in seq_along(n_cores_grid)) {
  cat("  Cores:", n_cores_grid[k], 
      "| Time:", round(par_times[k], 4), "s",
      "| Speedup vs serial opt:", round(time_opt["elapsed"] / par_times[k], 2), "x\n")
}

# Save the plot using pdf (works on any system, no X11 needed)
pdf("parallel_plot.pdf", width = 7, height = 5)
plot(n_cores_grid, par_times, type = "b", col = "blue", pch = 16, lwd = 2,
     xlab = "Number of Cores", ylab = "Elapsed Time (seconds)",
     main = "Problem 5(3): Running Time vs Number of Cores")
abline(h = time_opt["elapsed"], col = "red", lty = 2, lwd = 2)
legend("topright", legend = c("Parallel", "Serial optimized"),
       col = c("blue", "red"), lty = c(1, 2), pch = c(16, NA), lwd = 2)
dev.off()

cat("\nPlot saved to parallel_plot.pdf\n")
cat("\n=== ALL DONE ===\n")
