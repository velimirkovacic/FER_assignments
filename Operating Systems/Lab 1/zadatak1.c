#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <stdbool.h>

#define SLEEP_TIME 5


/* Globalne varijable */
const char* obrada_dat = "obrada.txt";
const char* status_dat = "status.txt";
int trenutni;


/* Funkcije za obradu signala */
void obradi_sigusr1(int sig);
void obradi_sigterm(int sig);
void obradi_sigint(int sig);

/* Funkcije za rad s datotekama */
int procitaj_status(void);
void upisi_obradu(int x);
void upisi_status(int x);

/* Funkcije za izracun podataka */
int izracun(int x);
int inverz(int y);
int otkri_trenutnog();


int main(void)
{
	struct sigaction act;

	/* 1. maskiranje signala SIGUSR1 */
	act.sa_handler = obradi_sigusr1; 	/* kojom se funkcijom signal obrađuje */
	sigemptyset(&act.sa_mask);
	sigaddset(&act.sa_mask, SIGTERM); 	/* blokirati i SIGTERM za vrijeme obrade */
	act.sa_flags = 0; 			/* naprednije mogućnosti preskočene */
	sigaction(SIGUSR1, &act, NULL); 	/* maskiranje signala preko sučelja OS-a */

	/* 2. maskiranje signala SIGTERM */
	act.sa_handler = obradi_sigterm;
	sigemptyset(&act.sa_mask);
	sigaction(SIGTERM, &act, NULL);
	
	/* 3. maskiranje signala SIGINT */
	act.sa_handler = obradi_sigint;
	sigaction(SIGINT, &act, NULL);

	printf("PID programa je %ld\n", (long) getpid());
	printf("Pocetak izvodenja\n");
	
	/* Glavni dio programa */
	trenutni = procitaj_status();
	
	if (trenutni == 0) {
		trenutni = otkri_trenutnog();
	}
	
	while (true) {
		trenutni++;
		upisi_status(0);
		upisi_obradu(izracun(trenutni));
		upisi_status(trenutni);
	}

	return 0;
}


/* Funkcija za obradu signala SIGINT */
void obradi_sigint(int sig)
{
	printf("SIGINT primljen\n");
	printf("Kraj izvodenja\n");
	exit(1);
}
	

/* Funkcija za obradu signala SIGTERM */
void obradi_sigterm(int sig)
{
	printf("SIGTERM primljen\n");
	upisi_status(trenutni);
	printf("Kraj izvodenja\n");
	exit(1);
}
	

/* Funkcija za obradu signala SIGUSR1 */
void obradi_sigusr1(int sig)
{
	printf("SIGUSR1 primljen\n");
	printf("Status je %d\n", trenutni);
}


int otkri_trenutnog(void)
{
	FILE *dat = fopen(obrada_dat, "r");
	int y;
	
	while (fscanf(dat, "%d", &y) == 1);
	
	fclose(dat);
	return inverz(y);
}


int izracun(int x)
{
	return x * x;
}	


int inverz(int y)
{
	int i;
	for (i = 1; izracun(i) != y; i++);

	return i;
}		


int procitaj_status(void)
{
	FILE *dat = fopen(status_dat, "r");
	int status;
	fscanf(dat, "%d", &status);
	fclose(dat);
	return status;
}


void upisi_status(int x)
{
	FILE *dat = fopen(status_dat, "w");
	fprintf(dat, "%d", x);
	fclose(dat);
}


void upisi_obradu(int x)
{
	FILE *dat = fopen(obrada_dat, "a");
	fprintf(dat, "%d\n", x);
	fclose(dat);
	
	for (int i = 1; i <= SLEEP_TIME; i++) {
		sleep(1);
	}
}
