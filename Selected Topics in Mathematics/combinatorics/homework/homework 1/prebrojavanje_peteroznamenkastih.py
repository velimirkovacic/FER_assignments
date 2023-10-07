
# Koliko ima peteroznamenkastih brojeva koji imaju najvise jednu znamenku 1 i 
# najvise jednu znamenku 2, a ako imaju obje, onda je znamenka 1 lijevo od znamenke 2?

def check(broj):
    broj = str(broj)
    return broj.count("1") <= 1 and broj.count("2") <= 1 and (broj.find("1") < broj.find("2") or (broj.find("2") == -1 or broj.find("1") == -1))


def main():
    brojac = 0
    for i in range(10000, 100000):
        brojac += check(i)

    print("Takvih brojeva ima: ", brojac)
    return


main()
