import objekti


def dfs(cvor, lista):
    if cvor.roditelji != None:
        dfs(cvor.roditelji[0], lista)
        dfs(cvor.roditelji[1], lista)
    if cvor not in lista:
        lista.append(cvor)


def ispis(klauzula1, klauzula2):
    lista = []
    dfs(objekti.Klauzula([objekti.Literal("NIL")], (klauzula1, klauzula2)), lista)
    indeks = 1
    
    print("="*80)

    for premis in lista:
        if premis.roditelji == None:
            print("{:<5} {:<60} {}".format(str(indeks) + ".", str(premis),'p',))
            premis.index = indeks
            indeks += 1

    for ded in lista:
        if ded.roditelji != None:
            print("{:<5} {:<60} {} i {}".format(str(indeks) + ".", str(ded), ded.roditelji[0].index, ded.roditelji[1].index))
            ded.index = indeks
            indeks += 1

    print("="*80)


def ciscenje(premise, sos, dodani):
    promjena = False
    stvarno_dodani = set()
    for klauzula1 in dodani:
        if klauzula1.tautologija():
            continue
        
        nastavi = False
        uklanjanje = set()
        for klauzula2 in premise:
            if klauzula1.issubset(klauzula2):
                uklanjanje.add(klauzula2)
            elif klauzula2.issubset(klauzula1):
                nastavi = True
                break
        if nastavi:
            continue

        premise = premise - uklanjanje
        
        uklanjanje = set()
        for klauzula2 in sos:
            if klauzula1.issubset(klauzula2):
                uklanjanje.add(klauzula2)
            elif klauzula2.issubset(klauzula1):
                nastavi = True
                break
        if nastavi:
            continue
        sos = sos - uklanjanje

        stvarno_dodani.add(klauzula1)
        promjena = True
    sos = sos | stvarno_dodani
    return premise, sos, promjena


def predciscenje(premise, sos):
    uklanjanje = set()
    for klauzula1 in premise | sos:
        if klauzula1.tautologija():
            uklanjanje.add(klauzula1)
            continue
        
        for klauzula2 in premise | sos:
            if klauzula1 != klauzula2 and klauzula1.issubset(klauzula2):
                uklanjanje.add(klauzula2)
                break
    sos = sos - uklanjanje
    premise = premise - uklanjanje


    return premise, sos


def rezolucija(premise, sos, ciljna):
    premise, sos = predciscenje(premise, sos)
    promjena = True
    while promjena:
        dodani = set()
        promjena = False
        
        for klauzula1 in sos:
            for klauzula2 in sos | premise:
                if klauzula2 in klauzula1.krizan:
                    continue
                else:
                    klauzula2.krizan.add(klauzula1)
                    klauzula1.krizan.add(klauzula2)

                l = klauzula1.rezolutni(klauzula2)
                if l:
                    novi = set(klauzula1.set | klauzula2.set)
                    novi.remove(l)
                    lneg = objekti.Literal(l.naziv, not l.negiran)
                    novi.remove(lneg)

                    if len(novi) == 0:
                        ispis(klauzula1, klauzula2)
                        print("[CONCLUSION]:", ciljna, "is true")
                        return True
                    
                    novi = objekti.Klauzula(novi, (klauzula1, klauzula2))

                    dodani.add(novi)

        premise, sos, promjena = ciscenje(premise, sos, dodani)
    
    print("[CONCLUSION]:", ciljna, "is unknown")
    return False


def kuhanje(klauzule, upiti):
    for upit in upiti:
        if upit.oznaka == "?":
            ciljna = upit.klauzula
            negirana_ciljna = set()
            
            for literal in ciljna.set:
                negirana_ciljna.add(objekti.Klauzula([objekti.Literal(literal.naziv, not literal.negiran)]))

            rezolucija(klauzule, negirana_ciljna, ciljna)
        elif upit.oznaka == "+":
            klauzule.add(upit.klauzula)
        elif upit.oznaka == "-" and upit.klauzula in klauzule:
            klauzule.remove(upit.klauzula)
