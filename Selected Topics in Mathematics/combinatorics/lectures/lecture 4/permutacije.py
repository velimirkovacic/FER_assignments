


brojac = 0

def permutiraj(a, trenutni_index, vrijednosti):
    
    if trenutni_index == a:
        global brojac
        brojac += 1

        print(" ".join([str(i) for i in vrijednosti]))
        return
        
    # print(vrijednosti)

    for i in range(1, a + 1):
        if i not in vrijednosti:
            vrijednosti[trenutni_index] = i
            permutiraj(a, trenutni_index + 1, vrijednosti)
    
    vrijednosti[trenutni_index] = 0
    return


def broj_permutacija(a):
    global brojac
    brojac = 0
    permutiraj(a, 0, [0] * a)
    return 



def main():
    a = int(input("|A| = "))

    A = [i for i in range(1, a + 1)]

    print("Skup A je:", A)
    broj_permutacija(a)

    global brojac
    print("Permutacija ima:", brojac)


main()