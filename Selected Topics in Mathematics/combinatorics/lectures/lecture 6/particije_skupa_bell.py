
# Particije n-clanog skupa prirodnih brojeva


def partitiraj(prvi, zadnji):
    if zadnji - prvi == 1:
        return [[[prvi, zadnji]], [[prvi], [zadnji]]]
    stare_particije = partitiraj(prvi + 1, zadnji)
    nove_particije = []

    for particija in stare_particije:
        nove_particije += [[[prvi]] + particija]

        for i in range(0, len(particija)):
            p = []
            j = 0
            for el in particija:
                if j == i:
                    p += [[prvi] + el]
                else:
                    p += [el]
                j += 1

            nove_particije += [p]

    return nove_particije


def ispis_particija(particije):
    for particija in particije:
        print(particija)


def broj_particija(n):
    particije = partitiraj(1, n)
    #ispis_particija(particije)
    return len(particije)


def main():
    a = int(input("|A| = "))

    A = [i for i in range(1, a + 1)]

    print("Skup A je:", A)
    

    print("Particije su:")
    broj = broj_particija(a)
    print("Particija ima / Bellov broj je: ", broj)

main()