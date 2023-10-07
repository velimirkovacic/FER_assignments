sim <- function() {
  kutija = c(rep('A', 70), rep('B', 30))
  izvlacenje = sample(kutija, 100, replace=FALSE)

  a = 0
  b = 0
  for(i in 1:100) {
    if(izvlacenje[i] == 'A') {
      a = a + 1
    } else {
      b = b + 1
    }
    
    if(b >= a) {
      return(0)
    }
  }
  return(1)
}



N = 50000
cnt = 0

for(i in 1:N) {
  cnt = cnt + sim()
}

# Rezultat:
print(cnt/N)

