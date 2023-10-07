
brojac = 0

def variraj(broj, pribr, trenutni_index, vrijednosti):
    
    if trenutni_index == pribr:
        if sum(vrijednosti) == broj:
            global brojac
            brojac += 1

            #print(" ".join([str(i) if i != 0 else "" for i in vrijednosti]))
        return
        
    # print(vrijednosti)
    prvi = vrijednosti[trenutni_index - 1]
    if prvi == 0:
        prvi = 1

    for i in range(prvi, broj + 1):
        vrijednosti[trenutni_index] = i
        if sum(vrijednosti) <= broj:
            variraj(broj, pribr, trenutni_index + 1, vrijednosti)
    
    vrijednosti[trenutni_index] = 0
    return

def broj_particija(broj, pribr):
    global brojac
    brojac = 0
    variraj(broj, pribr, 0, [0] * pribr)
    return 



def main():
    broj = int(input("Broj = "))
    pribr = int(input("Broj pribrojnika = "))

    broj_particija(broj, pribr)

    global brojac
    print("Particija ima:", brojac)


main()