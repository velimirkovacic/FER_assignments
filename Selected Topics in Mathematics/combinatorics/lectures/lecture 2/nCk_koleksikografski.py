
# Ispisuje sve k-clane podskupove n-clanog skupa. Koleksikografski poredak.

brojac = 0

def permutiraj(n, k, trenutni_index, vrijednosti):
    
    if trenutni_index == k:
        global brojac
        brojac += 1

        print(" ".join([str(i) for i in vrijednosti[:k]]))
        return
        
    # print(vrijednosti)
    if trenutni_index == 0:
        for i in range(vrijednosti[trenutni_index - 1], n + 1):
            vrijednosti[trenutni_index] = i
            permutiraj(n, k, trenutni_index + 1, vrijednosti)
    else:
        for i in range(1, vrijednosti[trenutni_index - 1]):
            vrijednosti[trenutni_index] = i
            permutiraj(n, k, trenutni_index + 1, vrijednosti)

    vrijednosti[trenutni_index] = 0


def podskupovi(n, k):
    global brojac
    brojac = 0
    
    permutiraj(n, k, 0, [0] * k + [k])
    return 



def main():
    n = int(input("n = "))
    k = int(input("k =  "))

    niz = [i for i in range(1, n + 1)]
    print("Niz je:", niz)
    podskupovi(n, k)

    global brojac
    print("Podskupova ima:", brojac)


main()