


brojac = 0

def variraj(n, suma, trenutni_index, vrijednosti):
    
    if trenutni_index == n:
        if sum(vrijednosti) <= suma:
            global brojac
            brojac += 1

            #print(" ".join([str(i) for i in vrijednosti]))
        return
        
    # print(vrijednosti)

    for i in range(0, suma + 1):
        vrijednosti[trenutni_index] = i
        if sum(vrijednosti) <= suma:
            variraj(n, suma, trenutni_index + 1, vrijednosti)
    
    vrijednosti[trenutni_index] = 0
    return


def broj_funkcija(n, suma):
    global brojac
    brojac = 0
    variraj(n, suma, 0, [0] * n)
    return 


def main():
    n = int(input("Broj sumanada = "))
    suma = int(input("Suma = "))

    broj_funkcija(n, suma)

    global brojac
    print("Kombinacija s ponavljanjem ima:", brojac)


main()