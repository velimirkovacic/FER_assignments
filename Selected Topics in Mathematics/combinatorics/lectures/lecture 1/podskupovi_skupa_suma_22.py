
# Koliko ima podskupova skupa {1,2,...,11} kojima je zbroje lemenata 22? 

def provjeri(niz, suma):
    return sum(niz) == suma


def prebroji(niz, suma):
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

        if provjeri(podniz, suma):
            brojac += 1
            #print(podniz)
    
    return brojac


def main():
    niz = [i for i in range(1, 12)]
    print("Niz je:", niz)
    print("Podnizova sa sumom 22 ima:", prebroji(niz, 22))


main()