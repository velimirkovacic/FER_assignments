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
#include <semaphore.h>
#include <signal.h>

#define TRA 7						// Trajanje vožnje vrtuljka u sekundama.
#define KAP 5						// Kapacitet vrtuljka.
#define REP for(int i = 0; i < KAP; i++)		// Ponavljanje onoliko puta koliki je kapacitet vrtuljka.
#define FOR(n) for(int i = 0; i < n; i++)		// Ponavljanje n puta.


int randNum(int min, int max);
void spavaj(int t);
void stvori_OSEMe();
void unisti_OSEMe();
void vrtuljak();
void posjetitelj();
void obradi_sigint(int sig);


sem_t* VRTULJAK_SPREMAN;
sem_t* POSJETITELJ_SJEO;
sem_t* VRTULJAK_ZAVRSIO;
sem_t* POSJETITELJ_SISAO;
long long pidMain;


// Funkcija generira nasumični cijelu broj u intervalu [min, max].
int randNum(int min, int max)
{
	return rand() % (max - min + 1) + min;
}



// Funkcija koja postavlja da proces spava t sekundi.
void spavaj(int t)
{
	FOR(t) sleep(1);
}



// Funkcija za stvaranje svih potrebih OSEM-ova.
void stvori_OSEMe() 
{
	int ID = shmget(IPC_PRIVATE, 4 * sizeof(sem_t), 0600);
	sem_t *p = shmat(ID, NULL, 0);
	
	VRTULJAK_SPREMAN = p;
	POSJETITELJ_SJEO = p + 1;
	VRTULJAK_ZAVRSIO = p + 2;
	POSJETITELJ_SISAO = p + 3;
	
	sem_init(VRTULJAK_SPREMAN, 1, 0);
	sem_init(POSJETITELJ_SJEO, 1, 0);
	sem_init(VRTULJAK_ZAVRSIO, 1, 0);
	sem_init(POSJETITELJ_SISAO, 1, 0);
	
	shmctl(ID, IPC_RMID, NULL);
}



// Funkcija za čišćenje memorije od svih stvorenih OSEM-ova.
void unisti_OSEMe()
{
	sem_destroy(VRTULJAK_SPREMAN);
	shmdt (VRTULJAK_SPREMAN);
	
	sem_destroy(POSJETITELJ_SJEO);
	shmdt (POSJETITELJ_SJEO);
	
	sem_destroy(VRTULJAK_ZAVRSIO);
	shmdt (VRTULJAK_ZAVRSIO);
	
	sem_destroy(POSJETITELJ_SISAO);
	shmdt (POSJETITELJ_SISAO);
}



// Funkcija koja simulira rad vrtuljka.
void vrtuljak()
{
	while(true) {
		printf("Vrtuljak SPREMAN za voŽnju!\n");
		
		REP sem_post(VRTULJAK_SPREMAN);
		REP sem_wait(POSJETITELJ_SJEO);	
		
		spavaj(1);	
		
		printf("Vrtuljak KREĆE s voŽnjom!\n");
		
		for(int i = 0; i < TRA; i++) {
			spavaj(1);
			printf("..<🎠>..\n");
		}
		
		printf("Vrtuljak ZAVRŠIO s vožnjom\n");
		
		spavaj(1);
		
		REP sem_post(VRTULJAK_ZAVRSIO);		
		REP sem_wait(POSJETITELJ_SISAO);
		
		spavaj(1);
	}
}



// Funkcija koja simulira rad posjetitelja.
void posjetitelj()
{
	srand(getpid());
	
	printf("Posjetitelj %ld ČEKA na vrtuljak.\n", (long) getpid());
	sem_wait(VRTULJAK_SPREMAN); 			
	
	spavaj(randNum(1, 6));
	
	printf("Posjetitelj %ld SJEO na vrtuljak.\n", (long) getpid());
	sem_post(POSJETITELJ_SJEO);	
	sem_wait(VRTULJAK_ZAVRSIO);				
	
	spavaj(randNum(1, 6));
	
	printf("Posjetitelj %ld SIŠAO s vrtuljka.\n", (long) getpid());
	sem_post(POSJETITELJ_SISAO);			
}



// Funkcija koja se poziva za SIGNAL_INTERRUPT. Briše memoriju i izlazi iz procesa.
void obradi_sigint(int sig)
{
	if(getpid() == pidMain) {
		printf("\nProgram prekinut, brišem memoriju.\n");
		unisti_OSEMe();
	}
	exit(0);
}



// Glavni program.
int main(void)
{	
	stvori_OSEMe();
	srand(time(NULL));
	
	pidMain = getpid();
	
	struct sigaction act;
	act.sa_handler = obradi_sigint;
	sigaction(SIGINT, &act, NULL);

	
	if(fork() == 0) {
		vrtuljak();
		exit(0);
	}
	
	while(true) {
		spavaj(randNum(1, 5));
		if(fork() == 0) {
			posjetitelj();
			exit(0);
		}
	}
	
	while(true) wait(NULL);

	unisti_OSEMe();
	return 0;
}
