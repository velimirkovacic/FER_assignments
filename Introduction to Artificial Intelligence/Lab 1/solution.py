
import sys
import getopt
import unos
import Pretrazivanja
import Provjere

# Izvor: https://opensourceoptions.com/blog/how-to-pass-arguments-to-a-python-script-from-the-command-line/#:~:text=The%20script%20will%20have%20the,argument%20name%20(%20%2D%2Dhelp%20).

opts, args = getopt.getopt(sys.argv[1:], "", ["alg=", "ss=", "h=", "check-optimistic", "check-consistent"])

algoritam = ""
putanja_opisa = ""
putajna_heuristike = ""
provjera_optimisticnosti = False
provjera_konzistentnosti = False

for opt in opts:
    if opt[0] == "--alg":
        algoritam = opt[1]

    elif opt[0] == "--ss":
        putanja_opisa = opt[1]
    
    elif opt[0] == "--h":
        putajna_heuristike = opt[1]

    elif opt[0] == "--check-optimistic":
        provjera_optimisticnosti = True
    
    elif opt[0] == "--check-consistent":
        provjera_konzistentnosti = True


opis = open(putanja_opisa, encoding="utf-8", mode="r")
ulaz = opis.read().split("\n")
opis.close()

poc, zav, stanja = unos.unos(ulaz)


if putajna_heuristike:
    opis = open(putajna_heuristike, encoding="utf-8", mode="r")
    ulaz = opis.read().split("\n")
    opis.close()
    unos.unos_heuristike(ulaz, stanja)

if provjera_optimisticnosti:
    Provjere.provjera_optimisicnosti(stanja, zav, putajna_heuristike)
elif provjera_konzistentnosti:
    Provjere.provjera_konzistentnosti(stanja, putajna_heuristike)
elif algoritam == "bfs":
    Pretrazivanja.BFS(poc, zav)
elif algoritam == "ucs":
    Pretrazivanja.UCS(poc, zav)
elif algoritam == "astar":
    Pretrazivanja.Astar(poc, zav)
