from typing import Any
import copy
import math

class Vrijednost:
    def __init__(self, vrijednost):
        self.vrijednost = vrijednost
    def __eq__(self, other):
        return isinstance(other, Vrijednost) and self.vrijednost == other.vrijednost
    def __hash__(self):
        return hash(self.vrijednost)
    def __lt__(self, other):
        return self.vrijednost < other.vrijednost
    def __repr__(self):
        return self.vrijednost
    
class ZnacajkaVrijednost:
    def __init__(self, znacajka, vrijednost):
        self.znacajka = znacajka
        self.vrijednost = vrijednost
    def __eq__(self, other):
        return isinstance(other, ZnacajkaVrijednost) and self.vrijednost == other.vrijednost and self.znacajka == other.znacajka
    def __hash__(self):
        return hash((self.znacajka, self.vrijednost))
    def __repr__(self):
        return str(self.znacajka) + " = " + str(self.vrijednost)

class Znacajka:
    def __init__(self, naziv):
        self.naziv = naziv
        self.mogucnosti = set()
    def dodajMogucnost(self, vrijednost):
        self.mogucnosti.add(vrijednost)
    def __eq__(self, other):
        return isinstance(other, Znacajka) and self.naziv == other.naziv
    def __hash__(self):
        return hash(self.naziv) #mogao bi biti problem
    def __repr__(self):
        return self.naziv
    
class Ciljna(Znacajka):
  pass

class Cvor:
    def __init__(self,  znacajka, dubina):
        self.znacajka = znacajka
        self.dubina = dubina
        #self.roditelj = roditelj
        self.djeca = dict()
    def dodajDijete(self, cvor, vrijednost):
        self.djeca[vrijednost] = cvor
    def __eq__(self, other):
        return isinstance(other, Cvor) and self.znacajka == other.znacajka and self.dubina == other.dubina #and self.roditelj == other.roditelj
    def __hash__(self):
        return hash((self.znacajka, self.dubina))#, self.roditelj))
    def ispis(self):
        self.obilazakIspis("")
    
    def obilazakIspis(self, ispis):
        ispis += str(self.dubina) + ":" + str(self.znacajka) 

        for vrijednost in sorted(self.djeca.keys()):
            if vrijednost == Vrijednost("$$$$$"): continue
            dijete = self.djeca[vrijednost]
            dijete.obilazakIspis(ispis + "=" + str(vrijednost) + " ")

class List:
    def __init__(self, vrijednost, dubina):
        self.vrijednost = vrijednost
        self.dubina = dubina
        #self.roditelj = roditelj
    def __eq__(self, other):
        return isinstance(other, List) and self.vrijednost == other.vrijednost and self.dubina == other.dubina #and self.roditelj == other.roditelj
    def __hash__(self):
        return hash((self.vrijednost, self.dubina)) #, self.roditelj))
    
    def obilazakIspis(self, ispis):
        print(ispis  + str(self.vrijednost))
    def ispis(self):
        print(self.vrijednost)

class Redak:
    def __init__(self, listaZnacajki, ciljna):
        self.znacajke = listaZnacajki
        self.ciljna = ciljna
    def __repr__(self):
        ispis = ""
        for znacajka in self.znacajke:
            ispis += str(znacajka.znacajka) + " = " + str(znacajka.vrijednost) + " "
        return ispis + str(self.ciljna.vrijednost)
    def __eq__(self, other):
        return isinstance(other, Redak) and self.znacajke == other.znacajke  and self.ciljna == other.ciljna

class Tablica:
    def __init__(self, retci):
        self.retci = retci
    def __repr__(self):
        ispis = ""
        for redak in self.retci:
            ispis += str(redak) + "\n"
        return ispis
    
    def __eq__(self, other):
        return isinstance(other, Tablica) and self.retci == other.retci

    def reduciraj(self, znacajkaVrijednost):
        retci = self.retci
        noviRetci = []

        for redak in retci:
            if znacajkaVrijednost in redak.znacajke:
                noviRedak = copy.deepcopy(redak)
                noviRedak.znacajke.remove(znacajkaVrijednost)
                noviRetci += [noviRedak]

        return Tablica(noviRetci)

    def velicina(self):
        return len(self.retci)
    
    def brojnostOznaka(self):
        brojnosti = dict()
        for redak in self.retci:
            ciljna = redak.ciljna
            if ciljna.vrijednost not in brojnosti.keys():
                brojnosti[ciljna.vrijednost] = 0
            brojnosti[ciljna.vrijednost] += 1
        return brojnosti
    
    def entropija(self):
        h = 0
        ukupno = self.velicina()
        brojnosti = self.brojnostOznaka()

        for vrijednost in brojnosti.keys():
            omjer = brojnosti[vrijednost] / ukupno
            h -= omjer * math.log2(omjer)

        return h
    
    def najcescaOznaka(self):
        brojnosti = self.brojnostOznaka()
        maxVrijednost = None
        maxBrojnost = 0
        for vrijednost in sorted(brojnosti.keys()):
            if brojnosti[vrijednost] > maxBrojnost:
                maxVrijednost = vrijednost
                maxBrojnost = brojnosti[vrijednost]

        return maxVrijednost

    def sviIshodiIsti(self, ozn):
        ukupno = self.velicina()
        brojnosti = self.brojnostOznaka()

        return ozn in brojnosti.keys() and brojnosti[ozn] == ukupno

    def brojZnacajki(self):
        return len(self.retci[0].znacajke)