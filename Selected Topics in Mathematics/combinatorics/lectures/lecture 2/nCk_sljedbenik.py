
# k-clani podskupovi n-clanog skupa. Funkcija sljedbenika. Leksikografski poredak.

def sljedbenik(n, k, podskup):
    index = k - 1
    maximal = n
    while index >= 0 and podskup[index] == n:
        index -= 1
        n -= 1
    if(index == 0 and podskup[index] == n or podskup[index] + 1 > n):
        return False
    else:
        podskup[index] += 1
        index += 1
        while index < k:
            podskup[index] = podskup[index - 1] + 1
            index += 1


def main():
    n = int(input("n = "))
    k = int(input("k =  "))

    podskup = [int(i) for i in input("Unesi podskup (leksikografski): ").split()] 


    niz = [i for i in range(1, n + 1)]
    print("Niz je:", niz)
    sljedbenik(n, k, podskup)

    print("Sljedbenik je: ", " ".join([str(i) for i in podskup]))


def main2():
    n = int(input("n = "))
    k = int(input("k =  "))

    niz = [i for i in range(1, n + 1)]
    print("Niz je:", niz)

    podskup = [i for i in range(1, k + 1)]
    print(" ".join([str(i) for i in podskup]))

    while sljedbenik(n, k, podskup) != False:
        print(" ".join([str(i) for i in podskup]))
    

main()