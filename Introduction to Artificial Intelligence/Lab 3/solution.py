import pomocne
import unos
import id3
import sys


args = sys.argv

putanjaUcenje = args[1]
putanjaProvjera = args[2]
maxDubina = None if len(args) <= 3 else int(args[3])


tablica = unos.unosTablice(putanjaUcenje)
testnaTablica = unos.unosTablice(putanjaProvjera)



id3 = id3.ID3.construct()

id3.fit(tablica, maxDubina)

id3.ispisStabla()

id3.predict(testnaTablica)

id3.accuracy(testnaTablica)

id3.confusion(testnaTablica)