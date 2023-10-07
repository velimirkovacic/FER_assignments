#python solution.py --train sine_train.txt --test sine_test.txt --nn 5s --popsiz e10 --elitism 1 --p 0.1 --K 0.1 --iter 10000

import sys
import csv
import mreza
from queue import PriorityQueue
import genetika

class Podatak:
    def __init__(self, ulaz, izlaz):
        self.ulaz = ulaz
        self.izlaz = izlaz


def main():
    args = sys.argv

    treniranje_dat = args[2]
    test_dat = args[4]
    neuroni = args[6]
    populacija = int(args[8])
    elitizam = int(args[10])
    p = float(args[12])
    K = float(args[14])
    iteracije = int(args[16])

    skupTreniranje = []
    skupTestiranje = []

    # Izvor: https://realpython.com/python-csv/
    with open(treniranje_dat) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row_raw in csv_reader:
            row = [float(i) for i in row_raw]
            skupTreniranje += [Podatak(row[:-1], row[-1])]

    with open(test_dat) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row_raw in csv_reader:
            row = [float(i) for i in row_raw]
            skupTestiranje += [Podatak(row[:-1], row[-1])]

    neuroni = [len(skupTreniranje[0].ulaz)] + [int(c) for c in neuroni.split("s")[:-1]] + [1]

    trenutnaPopulacija = []

    for i in range(populacija):
        trenutnaPopulacija += [mreza.Mreza(skupTreniranje, slojeviOp=neuroni)]



    for iterator in range(1, iteracije + 1):
        iducaPopulacija = sorted(trenutnaPopulacija)[:elitizam]

        if iterator % 2000 == 0:
            print("[Train error @" + str(iterator) + "]:", iducaPopulacija[0].err)

        while(len(iducaPopulacija) < populacija):
            m1 = genetika.proporcionalniOdabir(trenutnaPopulacija)
            m2 = genetika.proporcionalniOdabir(trenutnaPopulacija)
            while m1 == m2:
                m2 = genetika.proporcionalniOdabir(trenutnaPopulacija)

            m3 = genetika.krizanje(m1, m2, p, K, skupTreniranje)
            iducaPopulacija += [m3]

        trenutnaPopulacija = iducaPopulacija

    print("[Test error]:", sorted(trenutnaPopulacija)[0].error(skupTestiranje))



main()