
brojac = 0

def variraj(n, suma, trenutni_index, vrijednosti, domene):
    
    if trenutni_index == n:
        if sum(vrijednosti) <= suma:
            global brojac
            brojac += 1

            print(" ".join([str(i) for i in vrijednosti]))
        return
        
    # print(vrijednosti)

    for i in range(domene[trenutni_index][0], domene[trenutni_index][1] + 1):
        vrijednosti[trenutni_index] = i
        if sum(vrijednosti) <= suma:
            variraj(n, suma, trenutni_index + 1, vrijednosti, domene)
    
    vrijednosti[trenutni_index] = 0
    return


def broj_funkcija(n, suma, domene):
    global brojac
    brojac = 0
    variraj(n, suma, 0, [0] * n, domene)
    return 


def main():
    n = 6
    suma = 17
    domene = [(1, 4), (2, 4), (2, 4), (0, suma), (0, suma), (2, 4)] 

    broj_funkcija(n, suma, domene)

    global brojac
    print("Kombinacija s ponavljanjem ima:", brojac)


main()