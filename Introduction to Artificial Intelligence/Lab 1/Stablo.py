class Cvor:
    def __init__(self, stanje, roditelj = None, dubina = 0, cijena = 0):
        self.stanje = stanje
        self.roditelj = roditelj
        self.dubina = dubina
        self.cijena = cijena

    def __repr__(self):
        return str(self.stanje) + " " + str(self.dubina) + " " + str(self.cijena)
    
    def __lt__(self, other):
        
        if not self.stanje.ugasi_heuristiku and self.stanje.heuristika and self.stanje.heuristika + self.cijena != other.stanje.heuristika + other.cijena:
            return self.stanje.heuristika + self.cijena < other.stanje.heuristika + other.cijena
        
        if self.cijena != other.cijena:
            return self.cijena < other.cijena
        return self.stanje.naziv < other.stanje.naziv

class Stanje:
    def __init__(self, naziv):
        self.naziv = naziv
        self.susjedi = []
        self.heuristika = None
        self.ugasi_heuristiku = False

    def __repr__(self):
        return self.naziv

    def __lt__(self, other):
        return self.naziv < other.naziv