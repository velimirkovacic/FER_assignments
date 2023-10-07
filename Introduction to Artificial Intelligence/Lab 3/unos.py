from pomocne import *


def unosTablice(putanja):
    datoteka = open(putanja, mode="r")
    tablica_raw = datoteka.read().split("\n")
    datoteka.close()

    pozicijaZnacajka = dict()
    pozicija = 0
    for znacajka_naziv in tablica_raw[0].split(","):
        pozicijaZnacajka[pozicija] = Znacajka(znacajka_naziv)
        pozicija += 1

    retci = []
    for redak_raw in tablica_raw[1:]:
        if redak_raw == "":
            continue
        listaZnacajki = []
        pozicija = 0
        for vrijednost_naziv in redak_raw.split(","):
            vrijednost = Vrijednost(vrijednost_naziv)
            listaZnacajki += [ZnacajkaVrijednost(pozicijaZnacajka[pozicija], vrijednost)] 
            pozicijaZnacajka[pozicija].dodajMogucnost(vrijednost)
            pozicija += 1

        retci += [Redak(listaZnacajki[:-1], listaZnacajki[-1])]
    
    tablica = Tablica(retci)
    return tablica

