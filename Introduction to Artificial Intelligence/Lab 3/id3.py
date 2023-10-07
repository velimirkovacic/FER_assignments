from pomocne import *

class ID3:
    def __init__(self):
        pass



    def ispisStabla(self):
        print("[BRANCHES]:")
        self.vrsniCvor.ispis()



    def construct():
        return ID3()
            


    def fit(self, tablica, maxDubina):
        self.vrsniCvor = self.rekurzivniKonstruktor(tablica, None, 1, maxDubina)



    def rekurzivniKonstruktor(self, tablica, tablicaRoditelj, dubina, maxDubina = None):
        if tablica.velicina() == 0:
            ozn = tablicaRoditelj.najcescaOznaka()
            return List(ozn, dubina)
        
        ozn = tablica.najcescaOznaka()

        if tablica.brojZnacajki() == 0 or tablica.sviIshodiIsti(ozn) or (maxDubina != None and dubina > maxDubina):
            return List(ozn, dubina)

        diskZnacajka = None
        maxIg = -1

        for znacajkaVrijednost in tablica.retci[0].znacajke:
            znacajka = znacajkaVrijednost.znacajka

            ig = self.informacijskaDobit(tablica, znacajka)
            if ig > maxIg:
                maxIg = ig
                diskZnacajka = znacajka
        
        znacajka = diskZnacajka
        cvor = Cvor(znacajka, dubina)

        for vrijednost in znacajka.mogucnosti:
            cvor.dodajDijete(self.rekurzivniKonstruktor(tablica.reduciraj(ZnacajkaVrijednost(znacajka, vrijednost)), tablica, dubina + 1, maxDubina), vrijednost)
        
        cvor.dodajDijete(List(ozn, dubina + 1), Vrijednost("$$$$$"))

        return cvor



    def informacijskaDobit(self, tablica, znacajka):
        ig = tablica.entropija()

        for vrijednost in znacajka.mogucnosti:
            reduciranaTablica = tablica.reduciraj(ZnacajkaVrijednost(znacajka, vrijednost))

            ig -= reduciranaTablica.velicina()/tablica.velicina() * reduciranaTablica.entropija()

        return ig
    

    def obidi(self, cvor, listaZnacajki):
        if isinstance(cvor, List):
            return cvor.vrijednost
        else:
            znacajkaCvora = cvor.znacajka
            vrijednost = None
            for vrijednostZnacajka in listaZnacajki:
                if vrijednostZnacajka.znacajka == znacajkaCvora:
                    vrijednost = vrijednostZnacajka.vrijednost
                    break
            
            
            if vrijednost not in cvor.djeca.keys():
                return cvor.djeca[Vrijednost("$$$$$")].vrijednost
            
            
            return self.obidi(cvor.djeca[vrijednost], listaZnacajki)



    def predict(self, tablica):
        print("[PREDICTIONS]:", end="", sep="")
        for redak in tablica.retci:
            print(" ", self.obidi(self.vrsniCvor, redak.znacajke), end="", sep="")
        print()



    def accuracy(self, tablica):
        print("[ACCURACY]: ", end="", sep="")

        brojPogodaka = 0

        for redak in tablica.retci:
            brojPogodaka += redak.ciljna.vrijednost == self.obidi(self.vrsniCvor, redak.znacajke)
        
        print('{:.5f}'.format(brojPogodaka/tablica.velicina(), 5))



    def confusion(self, tablica):
        print("[CONFUSION_MATRIX]:")

        mogucnosti = sorted(tablica.retci[0].ciljna.znacajka.mogucnosti)

        matrica = dict()
        for mogucnost in mogucnosti:
            matrica[mogucnost] = dict()
            for mogucnost2 in mogucnosti:
                matrica[mogucnost][mogucnost2] = 0

        for redak in tablica.retci:
            matrica[redak.ciljna.vrijednost][self.obidi(self.vrsniCvor, redak.znacajke)] += 1

        for mogucnost in mogucnosti:
            for mogucnost2 in mogucnosti:
                print(matrica[mogucnost][mogucnost2], " ", sep= "", end="")
            print()
