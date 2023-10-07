import sys

# Globalne varijable

prijelazi = dict()
prihvatljiva_stanja = set() 

# Nepotrebne golbalne varijable
stanja = set() 
abeceda = set() 
znakovi_stoga = set()



def unesi_prijelaze(lines):
    for i in range(7, len(lines)):
        if lines[i] == '':
            break

        podaci = [i.split(',') for i in lines[i].split("->")]
        stanje = podaci[0][0]
        ulaz = podaci[0][1]
        znak_stoga = podaci[0][2]
        sljedeca = podaci[1][0]
        sljedeci_znak_stoga = podaci[1][1]

        if sljedeca[0] == '#':
            continue

        if stanje not in prijelazi.keys():
            prijelazi[stanje] = dict()
        
        if ulaz not in prijelazi[stanje].keys():
            prijelazi[stanje][ulaz] = dict()

        prijelazi[stanje][ulaz][znak_stoga] = (sljedeca, sljedeci_znak_stoga)



def ispisi_podatke():
    print(prihvatljiva_stanja)
    print(stanja)
    print(abeceda)

    for i in prijelazi:
        for j in prijelazi[i]:
            for k in prijelazi[i][j]:
                print(i, j, k, ' -> ', prijelazi[i][j][k])



def ispis_momenta(stog, trenutno_stanje):
    if len(stog) == 0:
        print(trenutno_stanje, '#$|', end="", sep="")
    else:
        print(trenutno_stanje, '#', ''.join(reversed(stog)), "|", end="", sep="")



def epsilon_prelazi(stog, trenutno_stanje, kraj_niza):
    dogodio_se_prijelaz = True

    while dogodio_se_prijelaz:
            dogodio_se_prijelaz = False
            try:
                par = prijelazi[trenutno_stanje]['$'][stog[len(stog) - 1]]
                stog.pop()
                trenutno_stanje = par[0]
                if list(par[1])[0] != '$':
                    stog.extend(reversed(list(par[1])))

                ispis_momenta(stog, trenutno_stanje)
                dogodio_se_prijelaz = True

                if kraj_niza and trenutno_stanje in prihvatljiva_stanja:
                    break
            except:
                break

    return trenutno_stanje



def obicno_prelazi(stog, znak, trenutno_stanje):
    try:
        par = prijelazi[trenutno_stanje][znak][stog.pop()]
        trenutno_stanje = par[0]
        if list(par[1])[0] != '$':
            stog.extend(reversed(list(par[1])))
        ispis_momenta(stog, trenutno_stanje)

    except:
        print("fail|", end="")
        return trenutno_stanje, True

    return trenutno_stanje, False



def simuliraj(pocetno_stanje, pocetni_znak_stoga, ulazni_niz):
    trenutno_stanje = pocetno_stanje
    stog = [pocetni_znak_stoga]

    ispis_momenta(stog, trenutno_stanje)

    trenutno_stanje = epsilon_prelazi(stog, trenutno_stanje, False)
    fail = False

    for i in range(0, len(ulazni_niz)):
        kraj_niza = (i == len(ulazni_niz) - 1)
        trenutno_stanje, fail = obicno_prelazi(stog, ulazni_niz[i], trenutno_stanje)
        if fail or (trenutno_stanje in prihvatljiva_stanja and kraj_niza):
            break
        trenutno_stanje = epsilon_prelazi(stog, trenutno_stanje, kraj_niza)

    if trenutno_stanje in prihvatljiva_stanja and not fail:
        print(1)
    else:
        print(0)
    


def main():
    #file_object = open('input.txt', 'r')
    #lines = file_object.read().split("\n")

    lines = sys.stdin.read().split("\n")

    # Unos podataka
    ulazni_nizovi = [i.split(',') for i in lines[0].split('|')]
    stanja.update(lines[1].split(','))
    abeceda.update(lines[2].split(','))
    znakovi_stoga.update(lines[3].split(','))
    if lines[4] != '':
        prihvatljiva_stanja.update(lines[4].split(','))
    pocetno_stanje = lines[5]
    pocetni_znak_stoga = lines[6]
    unesi_prijelaze(lines)

    #ispisi_podatke()
    for niz in ulazni_nizovi:
        simuliraj(pocetno_stanje, pocetni_znak_stoga, niz)

main()