
prijelaz <- function(tr, br) {
  if(tr == 4) {
    if(br > 0) {
      return (4);
    } else {
      return (3);
    }
  } else if(tr == 3) {
    if(br > 0) {
      return (4);
    } else {
      return (2);
    }
  } else if(tr == 2) {
    if(br == 0) {
      return (1);
    } else if(br == 1) {
      return (3);
    } else if(br >= 2) {
      return (4);
    }
  } else if(tr == 1) {
    if(br <= 3) {
      return (br + 1);
    } else {
      return (4);
    }
  } else return (0);
}

n = 100^2

trenutno_st = 4;

stanja = rep(0, n);
stanja[1] = trenutno_st

freq = rep(0, 4);
freq[trenutno_st] = 1 


cat("Grupa: ", trenutno_st, "\n")


for (val in 2: n)
{
  broj_nesr = rpois(1, 1/2);
  trenutno_st = prijelaz(trenutno_st, broj_nesr);
  stanja[val] = trenutno_st
  cat("Broj nesreca:", broj_nesr, "\n")
  cat("Grupa: ", trenutno_st, "\n")
  freq[trenutno_st] = freq[trenutno_st] + 1
  
}

plot(1:n, stanja, type="l")

freq = freq / n
print(freq)
