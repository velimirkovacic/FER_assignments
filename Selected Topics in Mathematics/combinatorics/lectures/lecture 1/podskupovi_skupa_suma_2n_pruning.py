
# Koliko ima podskupova skupa {1,2,...,n} kojima je zbroje lemenata 2n? 

import datetime


brojac = 0

def permutiraj(n, trenutni_index, vrijednosti):
    if sum(vrijednosti) >= 2 * n or trenutni_index == n:
        if sum(vrijednosti) == 2 * n:
            global brojac
            brojac += 1
            # print(vrijednosti)
        return
        
    # print(vrijednosti)

    for i in range(vrijednosti[trenutni_index - 1] + 1, n + 1):
        vrijednosti[trenutni_index] = i
        permutiraj(n, trenutni_index + 1, vrijednosti)

    vrijednosti[trenutni_index] = 0


def prebroji(n):
    global brojac
    brojac = 0
    permutiraj(n, 0, [0] * n)
    return brojac


def main():
    n = int(input("Unesi n: "))

    niz = [i for i in range(1, n + 1)]
    print("Niz je:", niz)
    print("Podnizova sa sumom 2n ima:", prebroji(n))


def main2():
    begin_time = datetime.datetime.now()
    n = 0
    while(1):
        begin_time = datetime.datetime.now()
        print("Podnizova sa sumom 2n, za niz duljine n =", n, ", ima:", prebroji(n))
        print("Izvodenje traje: ", datetime.datetime.now() - begin_time)
        n += 1


main2()