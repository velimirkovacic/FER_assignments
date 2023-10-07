
# Zadan je skup {0,1,...,n − 1}. 
# Postoji li neki njegov k-clani podskup {a1,a2,...,ak}
# tako da se u listi diferencija a1 − a2, a2 − a1, a1 − a3, a3 − a1, ... , ak − ak−1 
# racunatoj modulo n svaki od brojeva 1,2,...,n − 1 pojavi jednak broj puta?

svi_podskupovi = []
dvoclani_podskupovi = []
rjesenja = []

def permutiraj(n, k, trenutni_index, vrijednosti, skup):
    
    if trenutni_index == k:

        skup.append([i - 1 for i in vrijednosti])
        return
        
    # print(vrijednosti)

    for i in range(vrijednosti[trenutni_index - 1] + 1, n + 1):
        vrijednosti[trenutni_index] = i
        permutiraj(n, k, trenutni_index + 1, vrijednosti, skup)

    vrijednosti[trenutni_index] = 0


def podskupovi(n, k):
    global svi_podskupovi
    permutiraj(n, k, 0, [0] * k, svi_podskupovi)

    global dvoclani_podskupovi
    permutiraj(k, 2, 0, [0] * 2, dvoclani_podskupovi)

    return 


def provjeri(diferencije, n):
    for i in range(2, n):
        if diferencije[i - 1] != diferencije[i]:
            return False
    return True


def rjesnje(n):
    global rjesenja

    for svi in svi_podskupovi:
        diferencije = [0] * n
        for dvoclan in dvoclani_podskupovi:
            dif1 = svi[dvoclan[0]] - svi[dvoclan[1]]
            dif2 = svi[dvoclan[1]] - svi[dvoclan[0]]

            diferencije[dif1] += 1
            diferencije[dif2] += 1
        # print(diferencije)
        
        if provjeri(diferencije, n):
                rjesenja.append(svi)

            

def main():
    n = int(input("n = "))
    k = int(input("k =  "))

    niz = [i for i in range(0, n)]
    print("Niz je:", niz)
    
    podskupovi(n, k)

    rjesnje(n)

    print("Rjesenja su:")
    for i in rjesenja: 
        print(i)

    
main()