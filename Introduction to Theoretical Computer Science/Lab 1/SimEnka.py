
# Globalne varijable
import sys


trenutna_stanja = set()
prijelazi = dict()

# Nepotrebne golbalne varijable
stanja = set() 
prihvatljiva_stanja = set() 
abeceda = set() 



def obican_prijelaz(znak):
    global trenutna_stanja

    nova_stanja = []

    for stanje in trenutna_stanja:
        if stanje in prijelazi.keys() and znak in prijelazi[stanje].keys():
            nova_stanja.extend(prijelazi[stanje][znak])

    trenutna_stanja = set(nova_stanja)
    return



def epsilon_prijelaz():
    nova_stanja = []
    bio_prijelaz = False

    for stanje in trenutna_stanja:
        if stanje in prijelazi.keys() and '$' in prijelazi[stanje].keys():
            nova_stanja.extend(prijelazi[stanje]['$'])

            for j in prijelazi[stanje]['$']:
                if j not in trenutna_stanja:
                    bio_prijelaz = True

    trenutna_stanja.update(nova_stanja)
    return bio_prijelaz



def izvedi_automat(pocetno_stanje, ulazni_niz):
    global trenutna_stanja

    prodena_stanja = []

    trenutna_stanja = set()
    trenutna_stanja.add(pocetno_stanje)

    while(epsilon_prijelaz()):
        continue

    tmp = list(trenutna_stanja)
    tmp.sort()
    prodena_stanja.append(tmp)
    
    for znak in ulazni_niz:
        obican_prijelaz(znak)
        
        while(epsilon_prijelaz()):
            continue

        tmp = list(trenutna_stanja)
        tmp.sort()
        prodena_stanja.append(tmp)
    
    return prodena_stanja



def ukloni_newLine(lines):
    tmp = []
    for line in lines:
        tmp.append(line.strip())
    return tmp



def unesi_prijelaze(lines):
    for i in range(5, len(lines)):

        podaci = [i.split(',') for i in lines[i].split("->")]
        stanje = podaci[0][0]
        ulaz = podaci[0][1]
        sljedeca = podaci[1]

        if sljedeca[0] == '#':
            continue

        if stanje not in prijelazi.keys():
            prijelazi[stanje] = dict()
        
        prijelazi[stanje][ulaz] = sljedeca

    return
            


def ispisi_podatke():
    print(prihvatljiva_stanja)
    print(stanja)
    print(abeceda)

    for i in prijelazi:
        print(i, prijelazi[i])

    return



def main():

   # file_object = open('input.txt')
    lines = sys.stdin.readlines()

    lines = ukloni_newLine(lines)

    # Unos podataka
    ulazni_nizovi = [i.split(',') for i in lines[0].split('|')]
    stanja.update(lines[1].split(','))
    abeceda.update(lines[2].split(','))
    prihvatljiva_stanja.update(lines[3].split(','))
    pocetno_stanje = lines[4]
    unesi_prijelaze(lines)

    ispisi_podatke()

    for niz in ulazni_nizovi:
        rezultat = izvedi_automat(pocetno_stanje, niz)
        rezultat = "|".join([",".join(i) if len(i) > 0 else "#" for i in rezultat])
        print(rezultat)

    return



main()