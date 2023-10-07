lambda = 0.75
p0 = dpois(0, lambda) #P(N = 0), N ~ P(lambda)
p1 = dpois(1, lambda) #...
p2 = dpois(2, lambda)
p3 = dpois(3, lambda)

# a) Trazimo mat. prijel. vjer.
p = matrix(
  c(p0, p1, p2, p3, 1 - p1 - p2 - p3 - p0,
    p0, 0, p1, p2, 1 - p0 - p1 - p2,
    0, p0, 0, p1, 1 - p0 - p1,
    0, 0, p0, 0, 1 - p0,
    0, 0, 0, p0, 1 - p0
  ),
  
  nrow = 5,
  ncol = 5,        
  byrow = TRUE         
)

p

# c) Trazim stac. dist.

A = t(p) - diag(rep(1, 5))
A = rbind(A, rep(1, 5))
A

b = c(0, 0, 0, 0, 0, 1)
b

stdist = qr.solve(A, b)
stdist

# d) Trazimo prosj. cijenu

cijene = c(200, 250, 400, 600, 900)

proscij = stdist %*% cijene
proscij[1, 1]
