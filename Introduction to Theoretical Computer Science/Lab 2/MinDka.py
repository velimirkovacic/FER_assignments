from re import S
import sys



prijelazi = dict()
stanja = set()
prihvatljiva_stanja = set()
abeceda = set()
pocetno_stanje = None



def unesi_prijelaze(lines):
    for i in range(4, len(lines)):

        try:
            podaci = [i.split(',') for i in lines[i].split("->")]
            stanje = podaci[0][0]
            ulaz = podaci[0][1]
            sljedeca = podaci[1][0]
        except:
            return

        if stanje not in prijelazi.keys():
            prijelazi[stanje] = dict()
        
        prijelazi[stanje][ulaz] = sljedeca
    return



def ispisi_podatke():
    print(",".join(sorted(list(stanja))))
    print(",".join(sorted(list(abeceda))))
    print(",".join(sorted(list(prihvatljiva_stanja))))
    print(pocetno_stanje)

    for stanje in sorted(prijelazi.keys()):
        for ulaz in sorted(prijelazi[stanje].keys()):
            print(stanje, ',', ulaz, '->', prijelazi[stanje][ulaz], sep="")

    return



def dodaj_dohvatljiva(dohvatljiva_stanja):
    dodano_novo = False
    nova_dohvatljiva_stanja = set()

    for stanje in dohvatljiva_stanja:
        for znak in abeceda:
            novo_stanje = prijelazi[stanje][znak]
            if novo_stanje not in dohvatljiva_stanja and novo_stanje not in nova_dohvatljiva_stanja:
                nova_dohvatljiva_stanja.add(novo_stanje)
                dodano_novo = True
    
    dohvatljiva_stanja.update(nova_dohvatljiva_stanja)
    return dodano_novo



def izbaci_nedohvatljiva():
    global stanja
    global prihvatljiva_stanja
    dohvatljiva_stanja = set([pocetno_stanje])

    while(dodaj_dohvatljiva(dohvatljiva_stanja)):
        continue

    for stanje in stanja:
        if stanje not in dohvatljiva_stanja:
            del prijelazi[stanje]
    
    prihvatljiva_stanja = prihvatljiva_stanja - (stanja - dohvatljiva_stanja)
    stanja = dohvatljiva_stanja

    return



def stvori_tablicu():
    tablica = dict()
    for stanje in stanja:
        tablica[stanje] = True
    tmp = tablica.copy()

    for stanje in stanja:
        tablica[stanje] = tmp.copy()

    return tablica

def pronadi_ekvivalentna_stanja():
    tablica = stvori_tablicu()

    for stanje1 in prihvatljiva_stanja:
        for stanje2 in stanja:
            if stanje1 != stanje2 and stanje1 in prihvatljiva_stanja and stanje2 not in prihvatljiva_stanja:
                tablica[stanje1][stanje2] = False
                tablica[stanje2][stanje1] = False

    promjena = True
    while promjena:
        promjena = False
        for stanje1 in stanja:
            for stanje2 in stanja:
                if stanje1 != stanje2 and tablica[stanje1][stanje2]:
                    for znak in abeceda:
                        if not tablica[prijelazi[stanje1][znak]][prijelazi[stanje2][znak]]:
                            tablica[stanje1][stanje2] = False
                            tablica[stanje2][stanje1] = False
                            promjena = True
                            break
    for stanje in stanja:
        tablica[stanje][stanje] = False
    
    ekvivalencije = []
    for stanje1 in stanja:
        for stanje2 in stanja:
            if tablica[stanje1][stanje2] and set([stanje1, stanje2]) not in ekvivalencije:
                ekvivalencije.append(set([stanje1, stanje2]))
    
    return ekvivalencije



def eliminiraj_ekvivaletno(ekvstanja):
    for i in range(0, len(ekvstanja)):
        for j in range(0, len(ekvstanja)):
            if i != j and len(ekvstanja[i] & ekvstanja[j]) > 0:
                ekvstanja[i] |= ekvstanja[j]
                ekvstanja.pop(j)
                return True


def izbaci_ekvivalentna():
    global pocetno_stanje
    ekvstanja = pronadi_ekvivalentna_stanja()

    while(eliminiraj_ekvivaletno(ekvstanja)):
        continue
    
    translacija = dict()
    for stanje in stanja:
        translacija[stanje] = stanje

    for set_stanja in ekvstanja:
        rep_stanje = sorted(set_stanja)[0]
        for stanje in sorted(set_stanja)[1:]:
            translacija[stanje] = rep_stanje
            stanja.remove(stanje)
            del prijelazi[stanje]
            if stanje in prihvatljiva_stanja:
                prihvatljiva_stanja.remove(stanje)

    pocetno_stanje = translacija[pocetno_stanje]


    for stanje in prijelazi:
        for ulaz in prijelazi[stanje]:
            prijelazi[stanje][ulaz] = translacija[prijelazi[stanje][ulaz]]
    
    return



def main():
    file_object = open('input.txt')
    lines = file_object.read().split("\n")

    #lines = sys.stdin.read().split("\n")

    global pocetno_stanje
    global prihvatljiva_stanja

    # Unos podataka
    stanja.update(lines[0].split(','))
    abeceda.update(lines[1].split(','))
    prihvatljiva_stanja.update(lines[2].split(','))
    if list(prihvatljiva_stanja)[0] == '':
        prihvatljiva_stanja = set()
    pocetno_stanje = lines[3]
    unesi_prijelaze(lines)

    #ispisi_podatke()
    izbaci_nedohvatljiva()
    izbaci_ekvivalentna()
    ispisi_podatke()
    return



main()