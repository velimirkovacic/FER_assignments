g <- function(c) {
  return(exp(-0.002 * (c - 500))/(1 + exp(-0.002 * (c - 500))))
}

f <- function(c) {
  return(10000/c)
}

simulacija <- function(M, N, c) {
  sumaGostiju = 0
  trStanje = 0
  
  fc = f(c)
  gc = g(c)
  
  for(i in 1:M) {
    brojDospjelih = rpois(1, fc)
    novoStanje = min(100, rbinom(1, trStanje, gc) + brojDospjelih)
    
    sumaGostiju = sumaGostiju + novoStanje
    trStanje = novoStanje
    #print(trStanje)
  }
  
  return(sumaGostiju/M)
}


N = 100     # Broj soba
M = 100000  # Broj dana
c = 550      # cijena noæenja

statDist = simulacija(M, N, c)

###############################################################################




izracunajOcekivanje <- function(N, M, stac_dist) {
  sum = 0
  
  for(i in 1:(N + 1)){
    sum = sum + stac_dist[i] * (i - 1)
  }
  return(sum)
}

izracunajZaradu <- function(N, c, ocekivanje) {
  print(paste0("Zahtjev za izraèun zarade po cijeni: ", c))

  
  prosjecna_zarada = ocekivanje * c
  return(prosjecna_zarada)
}

generirajGrafZaradePoCijenama <- function(MIN, MAX, STEP, N, M) {
  
  zarade = rep(0, length(seq(MIN, MAX, STEP))) 
  j = 1
  
  for(i in seq(MIN, MAX, STEP)) {
    print(i)
    zarade[j] = izracunajZaradu(N, i, simulacija(M, N, i))
    j = j + 1
  }
  print(zarade)
  plot(x = seq(MIN, MAX, STEP), y = zarade, xlab = "Cijena noæenja", ylab = "Prosjeæna dnevna zarada", type="l", col="blue")
}






MIN = 1
MAX = 1000 # Maks cijena do koje testiramo
STEP = 50

generirajGrafZaradePoCijenama(MIN, MAX, STEP, N, M)
