
# Na koliko se nacina iz skupa S = {1,2,...,10} 
# moze izabrati podskup koji ne sadrzi dva uzastopna broja?

brojac = 0

def permutiraj(n, duljina, trenutni_index, vrijednosti):
    #print(vrijednosti)
    if(trenutni_index == duljina):
        global brojac
        brojac += 1
        print(vrijednosti)
        return

    if(trenutni_index == 0):
        pocetak = 1
    else:
        pocetak = vrijednosti[trenutni_index - 1] + 2

    for i in range(pocetak, n + 1):
        vrijednosti[trenutni_index] = i
        permutiraj(n, duljina, trenutni_index + 1, vrijednosti)


def prebroji(n):
    for duljina_podskupa in range(0, n):
        permutiraj(n, duljina_podskupa, 0, [0] * duljina_podskupa)


def main():
    prebroji(10)
    print(brojac)


main()