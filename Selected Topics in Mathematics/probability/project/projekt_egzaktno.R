g <- function(c) {
  return(exp(-0.002 * (c - 500))/(1 + exp(-0.002 * (c - 500))))
}

f <- function(c) {
  return(10000/c)
}


stvoriMatPrijelaza <- function(N, c) {
  P = matrix(
    rep(0, (N + 1)^2),
    nrow = N + 1,  
    ncol = N + 1,        
    byrow = TRUE         
  )
  
  fc = f(c)
  gc = g(c)
  
  for(i in 0:N) {
    sum = 0
    for(j in 0:(N - 1)) {
      for(k in 0:j) {
        P[i+1, j+1] = P[i+1, j+1] + dbinom(x = j-k, size = i, prob = gc) * dpois(x = k, lambda = fc)
      }
      sum = sum + P[i + 1, j + 1]
    }
    P[i + 1, N + 1] = 1 - sum
  }
  
  return(P)
} 

izracunajStatDist <- function(N, P) {
  A = t(P) - diag(rep(1, N + 1))
  A = rbind(A, rep(1, N + 1))

  b = c(rep(0, N + 1), 1)

  stdist = qr.solve(A, b)
  return(stdist)
}

izracunajOcekivanje <- function(N, stac_dist) {
  sum = 0

  for(i in 1:(N + 1)){
    sum = sum + stac_dist[i] * (i - 1)
  }
  return(sum)
}

izracunajZaradu <- function(N, c) {
  print(paste0("Zahtjev za izraèun zarade po cijeni: ", c))
  
  P = stvoriMatPrijelaza(N, c)
  
  stac_dist = izracunajStatDist(N, P)
  

  ocekivani_broj_izn = izracunajOcekivanje(N, stac_dist)

  prosjecna_zarada = ocekivani_broj_izn * c
  return(prosjecna_zarada)
}


generirajGrafZaradePoCijenama <- function(MIN, MAX, STEP, N) {
  
  zarade = rep(0, length(seq(MIN, MAX, STEP))) 
  j = 1

  for(i in seq(MIN, MAX, STEP)) {
    print(i)
    zarade[j] = izracunajZaradu(N, i)
    j = j + 1
  }
  print(zarade)
  plot(x = seq(MIN, MAX, STEP), y = zarade, xlab = "Cijena noæenja", ylab = "Prosjeæna dnevna zarada", type="l", col="blue")
}






MIN = 1
MAX = 1000 # Maks cijena do koje testiramo
STEP = 50
N = 100 # Broj soba


generirajGrafZaradePoCijenama(MIN, MAX, STEP, N)

# Maksimum je na 280.7

