import Pretrazivanja


def provjera_optimisicnosti(stanja, zavrsno, putanja):
    print("# HEURISTIC-OPTIMISTIC", putanja)

    neuspjeh = False

    for stanje in stanja:
        stanje.ugasi_heuristiku = True

    for stanje in sorted(stanja):
        heuristika = stanje.heuristika
        stvarna_cijena = Pretrazivanja.UCS(stanje, zavrsno, False) * 1.0

        print("[CONDITION]: [", end="")
        
        if (heuristika <= stvarna_cijena):
            print("OK", end="")
        else:
            print("ERR", end="")
            neuspjeh = True

        print("] h("+stanje.naziv+") <= h*: "+str(heuristika)+" <= "+str(stvarna_cijena))

    if neuspjeh:
        print("[CONCLUSION]: Heuristic is not optimistic.")
    else:
        print("[CONCLUSION]: Heuristic is optimistic.")


def provjera_konzistentnosti(stanja, putanja):

    neuspjeh = False

    print("# HEURISTIC-CONSISTENT", putanja)
    for stanje in sorted(stanja):
        for susjed in stanje.susjedi:
            susjedno_stanje = susjed[0]
            cijena = susjed[1]

            print("[CONDITION]: [", end="")
            
            if (stanje.heuristika <= susjedno_stanje.heuristika + cijena):
                print("OK", end="")
            else:
                print("ERR", end="")
                neuspjeh = True

            print("] h("+stanje.naziv+") <= h("+susjedno_stanje.naziv+") + c: "+str(stanje.heuristika)+" <= "+str(susjedno_stanje.heuristika) + " + " + str(cijena))


    if neuspjeh:
        print("[CONCLUSION]: Heuristic is not consistent.")
    else:
        print("[CONCLUSION]: Heuristic is consistent.")
