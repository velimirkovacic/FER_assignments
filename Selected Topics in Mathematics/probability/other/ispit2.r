sim <- function(N) {
  urna = c(rep(1, 14), rep(0, 100 - 14))
  
  for(i in 1:N) {
    kugl = sample(urna, 1)
    if(runif(1) <= 0.2) {
      j = 1
      while(urna[j] != kugl) {
        j = j + 1
      }
      urna[j] = !kugl
    }
  }
  return(sample(urna, 1))
}

N = 100000
vjer = 0

for(i in 0:N) {
  vjer = vjer + sim(100)
}

print(vjer/N)
