
lambda = 1/2
a0 = dpois(0, lambda)
a1 = dpois(1, lambda)
a2 = dpois(2, lambda)

P = matrix(
  c(a0, a1, a2, 1 - a0 - a1 - a2, 
    a0, 0, a1, 1 - a0 - a1, 
    0, a0, 0, 1 - a0,
    0, 0, a0, 1 - a0
    ),
  
  nrow = 4,  
  ncol = 4,        
  byrow = TRUE         
)


A = t(P) - diag(rep(1, 4))
A = rbind(A, rep(1, 4))
A

b = c(0, 0, 0, 0, 1)
b


qr.solve(A, b)


stac_dist_1 = qr.solve(A, b)
sum(stac_dist_1)
stac_dist_1

cijena = stac_dist_1 %*% c(200, 250, 400, 600)
cijena
