# 4 robota se natječu u streličarstvu. Zovu se A, B, C, D. Prvo gađa A, onda dođe B. 
# Ako B pogodi bliže od A ostaje u igri, inače ispada, itd.

import random

def simulacija_natjecanja():
    ispao = [0 for i in range(4)]
    zadnji_pogodak = 1.0
    u_igri = 4
    trenutni = 0

    while u_igri > 1:
        if not ispao[trenutni]:
            pogodak = random.uniform(0, 1)
            if pogodak > zadnji_pogodak:
                ispao[trenutni] = 1
                u_igri -= 1
            else:
                zadnji_pogodak = pogodak
        trenutni = (trenutni + 1) % 4
    return not ispao[1]


br = 0
n = 10000000

for i in range(n):
    br += simulacija_natjecanja()

print("Vjerojatnost da pobjedi 2. robot je:", br/n)