P = matrix(
  c(
    0.8, 0.2, 0, 0, 0,
    0.05, 0.8, 0.15, 0, 0,
    0, 0.1, 0.8, 0.1, 0,
    0, 0, 0.15, 0.8, 0.05,
    0, 0, 0, 0.2, 0.8
  ),
  
  nrow = 5,  
  ncol = 5,        
  byrow = TRUE         
)

A = t(P) - diag(rep(1, 5))
A = rbind(A, rep(1, 5))
A

b = c(0, 0, 0, 0, 0, 1)
b


qr.solve(A, b)


stac_dist_1 = qr.solve(A, b)
sum(stac_dist_1)
stac_dist_1

B = P%^%4
B

vc = 0
for(i in 1:5) {
  vc = vc + B[5, i] * (i - 1)/4
}
vc