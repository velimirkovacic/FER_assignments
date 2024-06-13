from mpi4py import MPI
from board import *
import sys
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# funkcija za obavljanje zadataka
def perform_tasks(tasks, depth):
    #print(rank, len(tasks))
    sys.stdout.flush()
    values = []
    for task in tasks:
        # gradi se stablo iz stanja i sprema vrijednost korijena
        t = Tree(task.board, depth, task.player)
        values.append(t.root.value)

    return values




if rank == 0:
    # procesor ranga 0 je voditelj, on raspodjeljuje zadatke
    search_depth = int(input("Search depth: "))                 # dubina stabla
    split_depth = int(input("Parellelization depth: "))         # dubina na kojoj se stvaraju zadaci
    rows = int(input("Rows: "))                                 # broj redaka tablice
    cols = int(input("Columns: "))                              # broj stupaca tablice

    b = Board(rows, cols)
    b.print_board()

    col = int(input("Player move: "))
    b = b.move( col, 1)
    b.print_board()
    sys.stdout.flush()

    player = 2  # trenutni igrac
    while(not b.check_winner( col) and not b.stalemate()):

        if player == 1:  # unos igracevog poteza
            movable = False
            while(not movable):
                col = int(input("Player move: "))
                movable = col >= 0 and col < b.cols and b.is_movable(col)

        else:   # racunanje poteza racunala

            start_time = time.time()
            t = Tree(b, split_depth)        # izgradnja stabla do dubine paralelizacije
            tasks = t.get_children()        # pribavljanje zadataka s najnize razine
            s = len(tasks) // size          # kolicina posla po procesoru


            # raspodjela poslova
            split = [s for i in range(size)]
            for i in range(1, len(tasks) - s * size + 1):
                split[i] += 1
            accum = split[0]
            for i in range(1, len(split)):
                prev = accum
                accum += split[i]
                comm.send((tasks[prev:accum], search_depth - split_depth), dest=i)
            
            values = perform_tasks(tasks[:split[0]], search_depth - split_depth)


            # pribavljanje rezultata
            for i in range(1, size):
                received_message = comm.recv(source=i)
                values.extend(received_message)

            # azuriranje vrijednosti stanja na dubini paralelizacije
            for i in range(len(tasks)):
                tasks[i].value = values[i]
            t.update_children()


            # odredivanje optimalnog poteza
            col = t.get_best_move()
            end_time = time.time()
            print("Computer move:", col)
            print("Elapsed time", end_time - start_time)

        # potez
        b = b.move(col, player)
        b.print_board()
        sys.stdout.flush()

        # promjena igraca
        player = 1 if player == 2 else 2

    # provjera kraja igre
    if b.stalemate():
        print("Stalemate")
    else:
        print("Winner = ", "computer" if b.check_winner(col) == 2 else "player")
    sys.stdout.flush()

    # poruka ostalim procesorima da zavrse s radom
    for j in range(1, size):
        comm.send(None, j)


# procesori radnici
else:
    while True:
        received_message = comm.recv(source=0)

        if received_message == None:  # poruka za zavrsetak rada
            break

        # racunanje i vracanje rezultata
        values = perform_tasks(received_message[0], received_message[1])
        comm.send(values, dest=0)





comm.Barrier()
MPI.Finalize()