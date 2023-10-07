
# Neka su A i B konacni skupovi, te neka je |A| = m i |B| = n. Koliko ima injektivnih funkcija iz A u B?
# Varijacije bez ponavljanja

brojac = 0

def variraj(a, b, trenutni_index, vrijednosti):
    
    if trenutni_index == a:
        global brojac
        brojac += 1

        print(" ".join([str(i) for i in vrijednosti]))
        return
        
    # print(vrijednosti)

    for i in range(1, b + 1):
        if i not in vrijednosti :
            vrijednosti[trenutni_index] = i
            variraj(a, b, trenutni_index + 1, vrijednosti)

    vrijednosti[trenutni_index] = 0
    return


def broj_funkcija(a, b):
    global brojac
    brojac = 0
    variraj(a, b, 0, [0] * a)
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
    print("Injektivnih funkcija ima:", brojac)


main()