
# Koliko ima 6-ˇclanih podskupova skupa {1,2,...,45} 
# takvih da sve razlike oblika ai − aj gledane modulo 45 budu razlicite?

svi_podskupovi = []
dvoclani_podskupovi = []
rjesenja = []

def permutiraj(n, k, trenutni_index, vrijednosti, skup):
    
    if trenutni_index == k:

        skup.append([i for i in vrijednosti])
        return
        
    # print(vrijednosti)

    for i in range(vrijednosti[trenutni_index - 1] + 1, n + 1):
        vrijednosti[trenutni_index] = i
        permutiraj(n, k, trenutni_index + 1, vrijednosti, skup)

    vrijednosti[trenutni_index] = 0


def podskupovi(n, k):
    global svi_podskupovi
    permutiraj(n, k, 0, [0] * k, svi_podskupovi)

    global dvoclani_podskupovi
    permutiraj(k, 2, 0, [0] * 2, dvoclani_podskupovi)

    return 


def provjeri(diferencije, n):

    for i in diferencije:
        if i > 1:
            return False
    return True


def rjesnje(n):
    global rjesenja

    for dvoclan in dvoclani_podskupovi:
        dvoclan[0] -= 1
        dvoclan[1] -= 1

    br = 1
    l = len(svi_podskupovi)
    print("Progress:")
    
    for svi in svi_podskupovi:
        if(br % 1000000 == 0):
            print(br // 1000000, "/", l // 1000000)
        br += 1

        diferencije = [0] * n
        
        for dvoclan in dvoclani_podskupovi:
            
            dif1 = svi[dvoclan[0]] - svi[dvoclan[1]]
            dif2 = svi[dvoclan[1]] - svi[dvoclan[0]]

            diferencije[dif1] += 1
            diferencije[dif2] += 1
        
        
        if provjeri(diferencije, n):
            rjesenja.append(svi)
            

def main():
    n = 45
    k = 6

    niz = [i for i in range(1, n + 1)]
    print("Niz je:", niz)
    
    podskupovi(n, k)
    
    rjesnje(n)

    print("Rjesenja ima: ", len(rjesenja))
    print("Leksikografski prvo rjesenje je: ", rjesenja[0])

 #   print("Rjesenja su:")
  #  for i in rjesenja: 
   #     print(i)


main()


