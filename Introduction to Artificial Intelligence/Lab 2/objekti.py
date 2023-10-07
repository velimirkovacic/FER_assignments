class Literal:
    def __init__(self, naziv, negiran=False):
        if naziv[0] == "~":
            naziv = naziv[1:]
            negiran = True

        self.naziv = naziv.strip(" ")
        self.negiran = negiran
    
    def __eq__(self, drugi):
        return self.naziv == drugi.naziv and self.negiran == drugi.negiran
    
    def __lt__(self, drugi):
        if self.naziv != drugi.naziv:
            return self.naziv < drugi.naziv
        else:
            return self.negiran < drugi.negiran
        
    def __str__(self):
        if self.negiran:
            return "~" + self.naziv
        return self.naziv
    
    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash((self.naziv, self.negiran))


class Klauzula:
    def __init__(self, znakovi, roditelji = None):
        self.set = set()
        self.roditelji = roditelji
        self.krizan = set()
        self.index = None


        for znak in znakovi:
            if isinstance(znak, Literal):
                self.set.add(znak)
            else:
                self.set.add(Literal(znak))
        self.set = frozenset(self.set)

    def rezolutni(self, drugi):
        suprotni = set()
        for literal in self.set:
            suprotni.add(Literal(literal.naziv, not literal.negiran))
        presjek = suprotni & drugi.set
        if len(presjek) == 1:
            for literal in presjek:
                return literal
        return False
    
    def issubset(self, drugi):
        return self.set.issubset(drugi.set)

    def tautologija(self):
        suprotni = set()
        for literal in self.set:
            suprotni.add(Literal(literal.naziv, not literal.negiran))
        
        return len(suprotni & self.set) > 0

    def __eq__(self, drugi):
        l = len(drugi.set & self.set)
        return l == len(self.set) and l == len(drugi.set)

    def __repr__(self):
        string = ""
        for literal in self.set:
            string += str(literal) + " v "

        return string[:-3]

    def __hash__(self):
        return hash(self.set)


class Upit:
    def __init__(self, klauzula, oznaka):
        self.klauzula = klauzula
        self.oznaka = oznaka

    def __repr__(self):
        return str(self.klauzula) + " ozn: " + self.oznaka