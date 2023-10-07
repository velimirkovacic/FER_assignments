# n ljudi ulazi u avion, svatko ima svoje sjedalo. 1. osoba koja ulazi sjeda na nasumično mjesto. 
# Svaka osoba kojoj nije slobodno vlastito mjesto sjeda na nasumično mjesto.
# Koja je vjerojatnost da zadnja osoba koja uđe može sjesti na svoje mjesto?

import numpy as np
import random

def simulacija_avionanja(n):
    sjedala = [0 for i in range(n)]
    putnici = np.random.permutation(n)
    sjedala[random.randrange(0, n - 1)] = 1

    for i in range(1, n-1):
        if(sjedala[putnici[i]]):
            k = random.randrange(0, n - 1)
            while(sjedala[k]):
                k = random.randrange(0, n - 1)
            sjedala[k] = 1
        else:
            sjedala[putnici[i]] = 1
    
    return not sjedala[putnici[n - 1]]


m = 1000000
n = 100
br = 0
for i in range(m):
    br += simulacija_avionanja(n)

print("Vjerojatnost da je zadnji putnik sjeo na svoje mjesto je:", br/m)