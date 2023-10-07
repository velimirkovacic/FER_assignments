
# Koliko ima podskupova skupa {1,2,...,n} kojima je zbroje lemenata 2n? 

import datetime

def provjeri(niz, suma):
    return sum(niz) == suma


def prebroji(niz):
    N = len(niz)
    brojac = 0

    for i in range(2**N):
        podniz = []

        tmp = i
        index = N - 1

        while tmp > 0:
            if tmp % 2 == 1: 
                podniz = [niz[index]] + podniz

            tmp //= 2
            index -= 1

        if provjeri(podniz, 2 * N):
            brojac += 1
            #print(podniz)
    
    return brojac


def main():
    n = int(input("Unesi n: "))

    niz = [i for i in range(1, n + 1)]
    print("Niz je:", niz)
    print("Podnizova sa sumom 2n ima:", prebroji(niz))


def main2():
    begin_time = datetime.datetime.now()
    n = 0
    niz = []
    while(1):
        begin_time = datetime.datetime.now()
        print("Podnizova sa sumom 2n, za niz duljine n =", n, ", ima:", prebroji(niz))
        print("Izvodenje traje: ", datetime.datetime.now() - begin_time)
        
        n += 1
        niz += [n]


main2()