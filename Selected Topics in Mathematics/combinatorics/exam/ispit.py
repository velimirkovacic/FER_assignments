
# Koliko ima particija broja 30, takvih da je najmanje pribrojnik >= 3, a najveći 14

brojac = 0

def variraj(broj, trenutni_index, vrijednosti):
    
    if trenutni_index == broj:
        if sum(vrijednosti) == broj:
            dobar = max(vrijednosti) <= 14
            for i in vrijednosti:
                if i in (1, 2) or i > 14:
                    dobar = False
                    break
            if dobar:
                global brojac
                brojac += 1

                print(" ".join([str(i) if i != 0 else "" for i in vrijednosti]))
        return
        

    for i in range(vrijednosti[trenutni_index - 1], broj + 1):
        vrijednosti[trenutni_index] = i
        if sum(vrijednosti) <= broj:
            variraj(broj, trenutni_index + 1, vrijednosti)
    
    vrijednosti[trenutni_index] = 0
    return

def broj_particija(broj):
    global brojac
    brojac = 0
    variraj(broj, 0, [0] * broj)
    return 



def main():
    broj = 30

    broj_particija(broj)

    global brojac
    print("Particija ima:", brojac)


main()