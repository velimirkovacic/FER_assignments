N = 10000

# Integral sam modificirao supstitucijom na papiru kako bih dobio konacne medje

X = runif(N, min = 0, max = 1)

Y = 1/X^2 * exp(sin((1 - X)/X) - (1 - X)/X)

parc_sum = cumsum(Y)
proc_n = parc_sum / (1:N)

plot(1:N, proc_n, type = "l")

g = function(x){
  return(1/x^2 * exp(sin((1 - x)/x) - (1 - x)/x))
}

num_val = integrate(g, lower = 0, upper = 1)$value

abline(num_val, 0, col = "red")


print(paste0("MC rez: ",mean(Y)))
print(paste0("Egz rez: ",num_val))
