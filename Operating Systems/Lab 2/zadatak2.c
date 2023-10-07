#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>


int *ZajVarUlazRadna;
int ZajVarRadnaIzlazna;
int memID;
char* datoteka = "datoteka.txt";


void* Ulazna(void *arg);
void* Radna(void *arg);
void* Izlazna(void *arg);



int main(int argc, char *argv[])
{
	int arg;
	if (sscanf (argv[1], "%d", &arg) != 1) {
    		printf("Neuspješan dohvat argumenta s komandne linije\n");
		exit(1);
	}

	memID = shmget(IPC_PRIVATE, sizeof(int), 0600);
	
   	if (memID == -1) {
      		printf("Neuspješno stvaranje zajedničke memorije\n");
      		exit(1);
	}


	ZajVarUlazRadna = (int *) shmat(memID, NULL, 0);
	*ZajVarUlazRadna = 0;
	
	pthread_t thrdID[3];


	if (fork() == 0) {
		if (pthread_create(&thrdID[0], NULL, Ulazna, &arg) != 0) {
			printf("Neuspjesno stvaranje ULAZNE DRETVE\n");
      			exit(1);
      		}
      		pthread_join(thrdID[0], NULL);
      		printf("Završena ULAZNA DRETVA\n");
      		
      		exit(0);
   	}
	
	if (fork() == 0) {
		if (pthread_create(&thrdID[1], NULL, Radna, &arg) != 0) {
			printf("Neuspjesno stvaranje RADNE DRETVA\n");
      			exit(1);
      		}
      		if (pthread_create(&thrdID[2], NULL, Izlazna, &arg) != 0) {
			printf("Neuspjesno stvaranje IZLAZNE DRETVE\n");
      			exit(1);
      		}
      		
      		pthread_join(thrdID[1], NULL);
      		printf("Završena RADNA DRETVA\n");
      		
      		pthread_join(thrdID[2], NULL);
      		printf("Završena IZLAZNA DRETVA\n");
      		
      		exit(0);
	}
	
	wait(NULL);
	wait(NULL);
	
	(void) shmdt((char *) ZajVarUlazRadna);
   	(void) shmctl(memID, IPC_RMID, NULL);

	return 0;
}



void* Ulazna(void *arg)
{
	printf("Pokrenuta ULAZNA DRETVA\n");

	for (int i = *((int*) arg); i > 0; i--) {
		int sleepyTime = 5;
		while (sleepyTime = sleep(sleepyTime));	

		srand(time(NULL));
		int podatak = rand() % 100 + 1;
		
		printf("\nULAZNA DRETVA: generiran broj %d\n", podatak);
				
		*ZajVarUlazRadna = podatak;
	}
}



void* Radna(void *arg)
{
	printf("Pokrenuta RADNA DRETVA\n");	
	
	for (int i = *((int*) arg); i > 0; i--) {
		int podatak;
		
		while ((podatak = *ZajVarUlazRadna) == 0);
		
		printf("RADNA DRETVA: dohvaćen broj %d i povećan na %d\n", podatak, podatak + 1);
		
		*ZajVarUlazRadna = 0;
		ZajVarRadnaIzlazna = podatak + 1;
	}
}



void* Izlazna(void *arg) 
{
	printf("Pokrenuta IZLAZNA DRETVA\n");
	
	for (int i = *((int*) arg); i > 0; i--) {
		int podatak;
		
		while ((podatak = ZajVarRadnaIzlazna) == 0);
		
		ZajVarRadnaIzlazna = 0;
		FILE *dat = fopen(datoteka, "a");
		fprintf(dat, "%d\n", podatak);
		fclose(dat);
		
		printf("IZLAZNA DRETVA: broj %d zapisan u %s\n", podatak, datoteka);
	}
}
