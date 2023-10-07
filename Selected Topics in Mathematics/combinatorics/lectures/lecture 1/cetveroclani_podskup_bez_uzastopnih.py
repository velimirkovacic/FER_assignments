
# Na koliko se nacina iz skupa S = {1,2,...,10}
# moze izabrati cetveroclani podskup koji ne sadrzi dva uzastopna broja?

def prebroji(n):
    brojac = 0

    for i in range(1, n + 1):
        for j in range(i + 2, n + 1):
            for k in range(j + 2, n + 1):
                for h in range(k + 2, n + 1):
                    brojac += 1
                    #print(i, j, k, h)

    return brojac


def main():
    print("Takvih ima:", prebroji(10))


main()