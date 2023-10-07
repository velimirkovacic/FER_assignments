
sim <- function() {
  urne = rep(0, 100)
  bacanja = floor(runif(200, 1, 101))
  broj_pogod = 0
  
  for(b in bacanja) {
    if(urne[b] == 0) {
      urne[b] = 1
      broj_pogod = broj_pogod + 1
      if(broj_pogod > 80) {
        return(0)
      }
    }
  }
  
  if(broj_pogod == 80) {
    return(1)
  } else {
    return(0)
  }
}

n = 100000
k = 0
for(i in 1:n) {
  k = k + sim();
}

print(k/n)