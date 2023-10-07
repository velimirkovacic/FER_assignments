import numpy as np

def sigma(x):
    return 1/(1 + np.exp(-x))

class Sloj:
    def __init__(self, brojNeurona, brojUlaza, tezine = None, zadnji = False):
        if not tezine:
            tezine = []
            for i in range(brojNeurona):
                redak = []
                for j in range(brojUlaza + 1):
                    redak += [np.random.normal(0, 0.01, 1)[0]]
                tezine += [redak]

        self.tezine = tezine
        self.brojNeurona = brojNeurona
        self.brojUlaza = brojUlaza
        self.zadnji = zadnji

    def izlaz(self, ulaz):
        izlaz = []
        for redak in self.tezine:
            izlazRetka = 0
            
            for i in range(len(redak) - 1):
                izlazRetka += redak[i] * ulaz[i]
            izlazRetka += redak[-1]        
            
            if not self.zadnji:
                izlazRetka = sigma(izlazRetka)
            
            izlaz += [izlazRetka]

        return izlaz


class Mreza:
    def __init__(self, skupPodataka, slojeviOp = None, slojevi=None):
        if not slojevi:
            slojevi = []

            ulaz = slojeviOp[0]
            for slojOp in slojeviOp[1:-1]:
                neuroni = slojOp
                slojevi += [Sloj(neuroni, ulaz)]
                ulaz = neuroni
            slojevi += [Sloj(slojeviOp[-1], ulaz, zadnji=True)]

        self.slojevi = slojevi
        self.err = self.error(skupPodataka)
        self.fitness = 1/(1 + self.err)
        
    def propagiraj(self, x):
        for sloj in self.slojevi:
            x = sloj.izlaz(x)
        return x[0]
    
    def error(self, skupPodataka):
        suma = 0
        for podatak in skupPodataka:
            suma += (podatak.izlaz - self.propagiraj(podatak.ulaz))**2
        return 1/len(skupPodataka) * suma
    
    def __lt__(self, other):
        return self.fitness > other.fitness
