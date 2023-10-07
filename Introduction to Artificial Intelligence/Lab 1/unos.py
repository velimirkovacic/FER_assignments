import sys
import Stablo

def unos(ulaz):
    cisti_ulaz = [redak for redak in ulaz if redak != "" and redak[0] != "#"]

    ulaz = cisti_ulaz

    pocetno_naziv = ulaz[0]

    if len(ulaz[1].split(" ")) > 1:
        zavrsno_naziv = ulaz[1].split(" ")
    else:
        zavrsno_naziv = [ulaz[1]]

    pocetno_stanje = None
    zavrsno_stanje = []

    stanja = []
    naziv_stanje = dict()

    for redak in ulaz[2:]:
        naziv = redak[:redak.find(":")]
        stanje = Stablo.Stanje(naziv)
        if stanje.naziv == pocetno_naziv:
            pocetno_stanje = stanje
        elif stanje.naziv in zavrsno_naziv:
            zavrsno_stanje += [stanje]

        naziv_stanje[naziv] = stanje
        stanja += [stanje]

    i = 0
    for redak in ulaz[2:]:
        glavno_stanje = stanja[i]
        i += 1
        susjedi = []
        
        if redak[redak.find(":") + 2:].strip(" ") == "":
            continue

        ostatak = redak[redak.find(":") + 2:].split(" ")

        for pod in ostatak:
            par = pod.split(",")
            naziv_susjeda = par[0]
            udaljenost_susjeda = float(par[1])

            susjedno_stanje = naziv_stanje[naziv_susjeda]

        
            susjedi += [(susjedno_stanje, udaljenost_susjeda)]
            susjedi.sort()
        glavno_stanje.susjedi = susjedi

    return pocetno_stanje, zavrsno_stanje, stanja


def unos_heuristike(ulaz, stanja):

    naziv_stanje = dict()

    for stanje in stanja:
        naziv_stanje[stanje.naziv] = stanje

    for redak in ulaz:
        if redak == "": continue

        naziv = redak[:redak.find(":")]
        vrijednost = float(redak[redak.find(":") + 1:])

        naziv_stanje[naziv].heuristika = vrijednost

