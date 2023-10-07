import Stablo
import queue


def printer(pronadeno_rjesenje, naziv_alg, posjecena_stanja = None, duljina_puta = None, zavrsni_cvor = None, heuristika = None):
    print("#", naziv_alg)


    if pronadeno_rjesenje:
        print("[FOUND_SOLUTION]: yes")
        print("[STATES_VISITED]:", posjecena_stanja)
        print("[PATH_LENGTH]:", duljina_puta + 1)
        print("[TOTAL_COST]:", zavrsni_cvor.cijena)
        
        putanja = zavrsni_cvor.stanje.naziv
        trenutni_cvor = zavrsni_cvor.roditelj
        while(trenutni_cvor):
            putanja = trenutni_cvor.stanje.naziv + " => " + putanja
            trenutni_cvor = trenutni_cvor.roditelj
        print("[PATH]:", putanja)


    else:
        print("[FOUND_SOLUTION]: no")







def BFS(pocetno_stanje, zavrsno_stanje):
    red = [Stablo.Cvor(pocetno_stanje)] # može biti i queue.Queue
    posjeceni = set()

    while(red):
        trenutni_cvor = red.pop(0)
        
        posjeceni.add(trenutni_cvor.stanje)
        
        if trenutni_cvor.stanje in zavrsno_stanje:
            printer(True, "BFS", len(posjeceni), trenutni_cvor.dubina, trenutni_cvor)
            return 

        djeca = trenutni_cvor.stanje.susjedi

        for dijete in djeca:
            if dijete[0] not in posjeceni:
                red += [Stablo.Cvor(dijete[0], trenutni_cvor, trenutni_cvor.dubina + 1, trenutni_cvor.cijena + dijete[1])]

    printer(False, "BFS")
    return





def UCS(pocetno_stanje, zavrsno_stanje, ispis = True):
    prioritetni_red = queue.PriorityQueue()
    prioritetni_red.put(Stablo.Cvor(pocetno_stanje))

    posjeceni = set()

    while(not prioritetni_red.empty()):
        trenutni_cvor = prioritetni_red.get()

        posjeceni.add(trenutni_cvor.stanje)
        
        if trenutni_cvor.stanje in zavrsno_stanje:
            if ispis:
                printer(True, "UCS", len(posjeceni), trenutni_cvor.dubina, trenutni_cvor)
            return trenutni_cvor.cijena

        djeca = trenutni_cvor.stanje.susjedi

        for dijete in djeca:
            if dijete[0] not in posjeceni:
                prioritetni_red.put(Stablo.Cvor(dijete[0], trenutni_cvor, trenutni_cvor.dubina + 1, trenutni_cvor.cijena + dijete[1]))

    if ispis:
        printer(False, "UCS")
    return





def Astar(pocetno_stanje, zavrsno_stanje):
    prioritetni_red = queue.PriorityQueue()
    prioritetni_red.put(Stablo.Cvor(pocetno_stanje))

    posjeceni = set()
    posjeceni_cvorovi = set()

    while(not prioritetni_red.empty()):
        trenutni_cvor = prioritetni_red.get()

        posjeceni.add(trenutni_cvor.stanje)
        posjeceni_cvorovi.add(trenutni_cvor)
        
        if trenutni_cvor.stanje in zavrsno_stanje:
            printer(True, "A*", len(posjeceni), trenutni_cvor.dubina, trenutni_cvor)
            return 

        djeca = trenutni_cvor.stanje.susjedi

        for dijete in djeca:

            # Uklanjanje čvora stanja ako je već posjećeno ali s većom cijenom do njega

            #  if ∃m'∈ closed such that state(m') = state(m) then 
            #       if g(m') < g(m) then continue 
            #       else remove(m',closed)

            cvorovi_za_odstrel = []
            nastavi = False

            for cvor in posjeceni_cvorovi:
                if cvor.stanje == dijete[0]:
                    if cvor.cijena < trenutni_cvor.cijena + dijete[1]:
                        nastavi = True
                        break
                    else:
                        cvorovi_za_odstrel += [cvor]
            
            for cvor in cvorovi_za_odstrel:
                posjeceni_cvorovi.remove(cvor)
                posjeceni.remove(cvor.stanje)

            if nastavi:
                continue

            # Uklanjanje čvora stanja ako je već u prioritetnom redu ali s većom cijenom do njega

            #  if ∃m'∈ open such that state(m') = state(m) then 
            #       if g(m') < g(m) then continue 
            #       else remove(m',open)


            cvorovi_za_spasenje = []
            while(not prioritetni_red.empty()):
                cvor = prioritetni_red.get()

                if cvor.stanje == dijete[0]:
                    if cvor.cijena < trenutni_cvor.cijena + dijete[1]:
                        cvorovi_za_spasenje += [cvor]
                        nastavi = True
                        break
                else:
                    cvorovi_za_spasenje += [cvor]
            

            for cvor in cvorovi_za_spasenje:
                prioritetni_red.put(cvor)

            if nastavi:
                continue


            # Inače insertiraj to stanje kao novi čvor

            prioritetni_red.put(Stablo.Cvor(dijete[0], trenutni_cvor, trenutni_cvor.dubina + 1, trenutni_cvor.cijena + dijete[1]))

    printer(False, "A*")
    return