import sys
import algoritmi
import objekti

args = sys.argv

algoritam_rezolucije = False
putanja_klauzula = ""
algoritam_kuhanja = False
putanja_kuharice = ""

if args[1] == "resolution":
    algoritam_rezolucije = True
    putanja_klauzula = args[2]
elif args[1] == "cooking":
    algoritam_kuhanja = True
    putanja_klauzula = args[2]
    putanja_kuharice = args[3]

opis = open(putanja_klauzula, encoding="utf-8", mode="r")
klauzule_raw = opis.read().split("\n")
opis.close()


if algoritam_rezolucije:
    klauzule = [objekti.Klauzula(redak.lower().split(" v ")) for redak in klauzule_raw if redak != "" and redak[0] != "#"]
    ciljna = klauzule[-1]
    klauzule = set(klauzule[:-1])
    negirana_ciljna = set()

    for literal in ciljna.set:
        negirana_ciljna.add(objekti.Klauzula([objekti.Literal(literal.naziv, not literal.negiran)]))
    
    algoritmi.rezolucija(klauzule, negirana_ciljna, ciljna)

elif algoritam_kuhanja:
    opis = open(putanja_kuharice, encoding="utf-8", mode="r")
    upiti_raw = opis.read().split("\n")
    opis.close()

    klauzule = set([objekti.Klauzula(redak.lower().split(" v ")) for redak in klauzule_raw if redak != "" and redak[0] != "#"])
    upiti = [objekti.Upit(objekti.Klauzula(redak[:-1].lower().split(" v ")), redak[-1].lower()) for redak in upiti_raw if redak != "" and redak[0] != "#"]

    algoritmi.kuhanje(klauzule, upiti)
