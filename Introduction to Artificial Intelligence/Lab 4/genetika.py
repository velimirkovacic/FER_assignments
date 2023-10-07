import mreza
import random
import numpy as np

def sredina(x, y):
    return (x + y)/2

def proporcionalniOdabir(skupMreza):
    ukupno = 0
    for mreza in skupMreza:
        ukupno += mreza.fitness
    vjerojatnosti = []
    for mreza in skupMreza:
        vjerojatnosti += [mreza.fitness/ukupno]
    return skupMreza[np.random.choice(len(skupMreza), p=vjerojatnosti)]

        
        
def krizanje(m1, m2, p, K, skupPodataka):
    slojevi = []
    for i in range(len(m1.slojevi)):
        tezine = []
        for j in range(len(m1.slojevi[i].tezine)):
            redak = []
            for k in range(len(m1.slojevi[i].tezine[j])):
                tezina = sredina(m1.slojevi[i].tezine[j][k], m2.slojevi[i].tezine[j][k])
                if random.uniform(0, 1) < p:
                    tezina += np.random.normal(0, K, 1)[0]
                redak += [tezina]
            tezine += [redak]
        sloj = mreza.Sloj(brojNeurona = m1.slojevi[i].brojNeurona, brojUlaza= m1.slojevi[i].brojUlaza, tezine = tezine, zadnji = (i == len(m1.slojevi) - 1))
        slojevi += [sloj]
    return mreza.Mreza(skupPodataka, slojevi=slojevi)