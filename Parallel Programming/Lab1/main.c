#include<mpi.h>
#include<stdio.h>
#include<stdbool.h>
#include <time.h>
#include <stdlib.h>
#include<windows.h>

#define LIJEVA 0
#define DESNA 1

#define ZAHTJEV 10
#define DAVANJE 20

int main(int argc, char** argv) {

	//InitializetheMPI environment 
	MPI_Init(NULL, NULL);
	//Getthenumberofprocesses 
	int N;
	MPI_Comm_size(MPI_COMM_WORLD, &N);
	//Gettherankoftheprocess 
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//Getthenameoftheprocessor 
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	//Printoffahelloworldmessage 
	printf("Processor %s, rank %d out of %d processors\n", processor_name, rank, N);
	//FinalizetheMPIenvironment.
	fflush(stdout);
	Sleep(500);
	int ima[] = {false, false};

	int cisto[] = { false, false };

	int trazen[] = { false, false };

	if (rank == 0) {
		ima[LIJEVA] = true;
		ima[DESNA] = true;
	} 
	else if (rank != N - 1) {
		ima[LIJEVA] = true;
	}


	while (true) {

		srand(time(NULL) + rank);

		int r = rand() % 10 + 5;

		for (int k = 0; k < rank; k++) printf("\t\t\t");
		printf("mislim\n");
		//printf("Filozof %d pocinje razmisljati.\n", rank);
		fflush(stdout);
		for (int i = 0; i < r; i++) {
			Sleep(500);
			MPI_Status status;
			int flag = 0;
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);

			if (flag) {
				int poruka = 0;
				MPI_Recv(&poruka, 1, MPI_INT, status.MPI_SOURCE, ZAHTJEV, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//printf("Filozof %d dobio zahtjev od filozofa %d za vilicu.\n", rank, status.MPI_SOURCE);
				//fflush(stdout);
				if (poruka == LIJEVA && ima[DESNA] && !cisto[DESNA]) {
					ima[DESNA] = false;
					//printf("Filozof %d daje svoju desnu vilicu filozofu %d.\n", rank, status.MPI_SOURCE);
					//fflush(stdout);
					MPI_Send(&poruka, 1, MPI_INT, status.MPI_SOURCE, DAVANJE, MPI_COMM_WORLD);
	
				}
				else if (poruka == DESNA && ima[LIJEVA] && !cisto[LIJEVA]) {
					ima[LIJEVA] = false;
					MPI_Send(&poruka, 1, MPI_INT, status.MPI_SOURCE, DAVANJE, MPI_COMM_WORLD);
					//printf("Filozof %d daje svoju lijevu vilicu filozofu %d.\n", rank, status.MPI_SOURCE);
					//fflush(stdout);
				}

			}
			else {
				continue;
			}
		}
		/*printf("Filozof %d je gotov s razmisljanjem.\n", rank);*/

		while (!(ima[LIJEVA] && ima[DESNA])) {
			int trazena = LIJEVA;
			if (!ima[LIJEVA]) {
				int poruka = LIJEVA;
				/*printf("filozof %d trazi desnu vilicu filozofa %d.\n", rank, (rank + 1) % n);
				fflush(stdout);*/
				for (int k = 0; k < rank; k++) printf("\t\t\t");
				printf("trazim lijevu vilicu\n");
				fflush(stdout);
				MPI_Send(&poruka, 1, MPI_INT, (rank + 1) % N, ZAHTJEV, MPI_COMM_WORLD);
			}
			else if (!ima[DESNA]) {
				int poruka = DESNA;
				/*printf("Filozof %d trazi lijevu vilicu filozofa %d.\n", rank, (rank - 1 + N) % N);
				fflush(stdout);*/
				for (int k = 0; k < rank; k++) printf("\t\t\t");
				printf("trazim desnu vilicu\n");
				fflush(stdout);
				MPI_Send(&poruka, 1, MPI_INT, (rank - 1 + N) % N, ZAHTJEV, MPI_COMM_WORLD);
				trazena = DESNA;
			}

			do {
				int poruka = 0;
				MPI_Status status;
				MPI_Recv(&poruka, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

				if (status.MPI_TAG == ZAHTJEV) {
					/*printf("Filozof %d prima zahtjev od filozofa %d za vilicu.\n", rank, status.MPI_SOURCE);
					fflush(stdout);*/
					if (poruka == LIJEVA && ima[DESNA] && !cisto[DESNA]) {
						ima[DESNA] = false;
						/*printf("Filozof %d daje svoju desnu vilicu filozofu %d.\n", rank, status.MPI_SOURCE);
						fflush(stdout);*/

						MPI_Send(&poruka, 1, MPI_INT, status.MPI_SOURCE, DAVANJE, MPI_COMM_WORLD);
					}
					else if (poruka == DESNA && ima[LIJEVA] && !cisto[LIJEVA]) {
						ima[LIJEVA] = false;
						/*printf("Filozof %d daje svoju lijevu vilicu filozofu %d.\n", rank, status.MPI_SOURCE);
						fflush(stdout);*/

						MPI_Send(&poruka, 1, MPI_INT, status.MPI_SOURCE, DAVANJE, MPI_COMM_WORLD);
					}
					else {
						trazen[!poruka] = true;
						/*printf("Filozof %d pamti zahtjev za vilicom od filozofa %d.\n", rank, status.MPI_SOURCE);
						fflush(stdout);*/
					}
				}
				else if (status.MPI_TAG == DAVANJE) {
					/*printf("Filozof %d prima vilicu %d od filozofa %d.\n", rank, zahtjev, status.MPI_SOURCE);
					fflush(stdout);*/
					ima[poruka] = true;
					cisto[poruka] = true;
				}



			} while (!ima[trazena]);
		}

		r = rand() % 10 + 5;

		for (int k = 0; k < rank; k++) printf("\t\t\t");
		printf("jedem\n");
		fflush(stdout);


		for (int i = 0; i < r; i++) {
			Sleep(500);

		}

		/*printf("Filozof %d je gotov s jelom.\n", rank);
		fflush(stdout);*/

		cisto[LIJEVA] = false;
		cisto[DESNA] = false;

		if (trazen[LIJEVA]) {
			trazen[LIJEVA] = false;
			ima[LIJEVA] = false;
			int poruka = DESNA;
			/*printf("Filozof %d daje svoju desnu vilicu filozofu %d.\n", rank, (rank + 1 + N) % N);
			fflush(stdout);*/
			MPI_Send(&poruka, 1, MPI_INT, (rank + 1 + N) % N, DAVANJE, MPI_COMM_WORLD);
		}
		if (trazen[DESNA]) {
			trazen[DESNA] = false;
			ima[DESNA] = false;
			int poruka = LIJEVA;
			/*printf("Filozof %d daje svoju lijevu vilicu filozofu %d.\n", rank, (rank + 1 + N) % N);
			fflush(stdout);*/
			MPI_Send(&poruka, 1, MPI_INT, (rank - 1 + N) % N, DAVANJE, MPI_COMM_WORLD);
		}

	}

	MPI_Finalize();

	return 0;
}