
# Neka je A = {1,2,...,m} i B = {1,2,...,n}. Koliko ima surjektivnih funkcija iz A u B?

brojac = 0

def svi_iskoristeni(lista, velicina):
    for i in range(1, velicina + 1):
        if lista[i] == 0:
            return False
    return True


def variraj(a, b, trenutni_index, vrijednosti, iskoristeno):
    
    if trenutni_index == a:
        if svi_iskoristeni(iskoristeno, b):
            global brojac
            brojac += 1

           # print(" ".join([str(i) for i in vrijednosti]))
        return
        
    # print(vrijednosti)

    for i in range(1, b + 1):
        vrijednosti[trenutni_index] = i
        iskoristeno[i] += 1
        variraj(a, b, trenutni_index + 1, vrijednosti, iskoristeno)
        iskoristeno[i] -= 1



def broj_funkcija(a, b):
    global brojac
    brojac = 0
    variraj(a, b, 0, [0] * a, [0] * (b + 1))
    return 



def main():
    a = int(input("|A| = "))
    b = int(input("|B| = "))

    A = [i for i in range(1, a + 1)]
    B = [i for i in range(1, b + 1)]

    print("Skup A je:", A)
    print("Skup B je:", B)
    broj_funkcija(a, b)

    global brojac
    print("Surjektivnih funkcija ima:", brojac)


main()