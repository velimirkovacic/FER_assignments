
# Generiranje svih podskupova skupa u leksickom (binarnom) poretku.

def prirodni_leks(niz, N):
    for i in range(2**N):
        podniz = []

        tmp = i
        index = N - 1

        while tmp > 0:
            if tmp % 2 == 1: 
                podniz = [niz[index]] + podniz

            tmp //= 2
            index -= 1

        print(podniz)

def main():
    niz = [1, 2, 3]
    prirodni_leks(niz, len(niz))


main()