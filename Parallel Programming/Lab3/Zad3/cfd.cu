#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "arraymalloc.h"
#include "boundary.h"
#include "jacobi.h"
#include "cfdio.h"

__global__ void jacobistep_kernel(double *psinew, double *psi, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
 	int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < n && col < m) {
		int i = row + 1;
		int j = col + 1;

		psinew[i*(m+2)+j]=0.25*(psi[(i-1)*(m+2)+j]+psi[(i+1)*(m+2)+j]+psi[i*(m+2)+j-1]+psi[i*(m+2)+j+1]);
    }
}


__global__ void copy_kernel(double *psinew, double *psi, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
 	int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < n && col < m) {
		int i = row + 1;
		int j = col + 1;

		psinew[i*(m+2)+j]=psi[i*(m+2)+j];
    }
    
}


__global__ void deltasq_kernel(double *newarr, double *oldarr, double *result, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
 	int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < n && col < m) {
		int i = row + 1;
		int j = col + 1;
		double tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
        atomicAdd(result, tmp * tmp);
    }
}



int main(int argc, char **argv)
{
	int printfreq=1000; //output frequency
	double error;
	double  bnorm;
	double tolerance=0.0; //tolerance for convergence. <=0 means do not check

	//main arrays
	double *psi;
	//temporary versions of main arrays
	double *psitmp;

	//command line arguments
	int scalefactor, numiter;

	//simulation sizes
	int bbase=10;
	int hbase=15;
	int wbase=5;
	int mbase=32;
	int nbase=32;

	int irrotational = 1, checkerr = 0;

	int m,n,b,h,w;
	int iter;
	int i,j;

	double tstart, tstop, ttot, titer;

	//do we stop because of tolerance?
	if (tolerance > 0) {checkerr=0;}

	//check command line parameters and parse them

	if (argc <3|| argc >4) {
		printf("Usage: cfd <scale> <numiter>\n");
		return 0;
	}

	scalefactor=atoi(argv[1]);
	numiter=atoi(argv[2]);

	if(!checkerr) {
		printf("Scale Factor = %i, iterations = %i\n",scalefactor, numiter);
	}
	else {
		printf("Scale Factor = %i, iterations = %i, tolerance= %g\n",scalefactor,numiter,tolerance);
	}

	printf("Irrotational flow\n");

	//Calculate b, h & w and m & n
	b = bbase*scalefactor;
	h = hbase*scalefactor;
	w = wbase*scalefactor;
	m = mbase*scalefactor;
	n = nbase*scalefactor;

	printf("Running CFD on %d x %d grid in serial\n",m,n);

	//allocate arrays
	psi    = (double *) malloc((m+2)*(n+2)*sizeof(double));
	psitmp = (double *) malloc((m+2)*(n+2)*sizeof(double));

	//zero the psi array
	for (i=0;i<m+2;i++) {
		for(j=0;j<n+2;j++) {
			psi[i*(m+2)+j]=0.0;
		}
	}

	//set the psi boundary conditions
	boundarypsi(psi,m,n,b,h,w);

	//compute normalisation factor for error
	bnorm=0.0;

	for (i=0;i<m+2;i++) {
			for (j=0;j<n+2;j++) {
			bnorm += psi[i*(m+2)+j]*psi[i*(m+2)+j];
		}
	}
	bnorm=sqrt(bnorm);



	// CUDA DIO
	dim3 blockDim(32, 32);
	dim3 gridDim(m/32, n/32);

	double *d_psi;
	double *d_psitmp;
	double *d_result;
	double *h_result;

 	h_result = (double*)malloc(sizeof(double));
	*h_result = 0.0f;

	
	cudaMalloc(&d_psi, (m+2) * (n+2) * sizeof(double));
	cudaMalloc(&d_psitmp, (m+2) * (n+2) * sizeof(double));
	cudaMalloc(&d_result, sizeof(double));

	cudaMemcpy(d_psi, psi, (m+2) * (n+2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_psitmp, psitmp, (m+2) * (n+2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);

	// KRAJ CUDA DIJELA



	//begin iterative Jacobi loop
	printf("\nStarting main loop...\n\n");
	tstart=gettime();

	for(iter=1;iter<=numiter;iter++) {

		//calculate psi for next iteration
		jacobistep_kernel<<<gridDim, blockDim>>>(d_psitmp, d_psi, m, n); // CUDA
	
		//calculate current error if required
		if (checkerr || iter == numiter) {

			*h_result = 0.0f;

			cudaMemcpy(d_result, h_result, sizeof(double), cudaMemcpyHostToDevice);
			deltasq_kernel<<<gridDim, blockDim>>>(d_psitmp, d_psi, d_result, m,n); // CUDA
			cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost); // CUDA
			
			error = *h_result;

			error=sqrt(error);
			error=error/bnorm;

		}
		//quit early if we have reached required tolerance
		if (checkerr) {
			if (error < tolerance) {
				printf("Converged on iteration %d\n",iter);
				break;
			}
		}

		//copy back
		copy_kernel<<<gridDim, blockDim>>>(d_psi, d_psitmp, m, n); // CUDA

		//print loop information
		if(iter%printfreq == 0) {
			if (!checkerr) {
				printf("Completed iteration %d\n",iter);
			}
			else {
				printf("Completed iteration %d, error = %g\n",iter,error);
			}
		}
	}	// iter

	if (iter > numiter) iter=numiter;

	tstop=gettime();

	ttot=tstop-tstart;
	titer=ttot/(double)iter;

	//print out some stats
	printf("\n... finished\n");
	printf("After %d iterations, the error is %g\n",iter,error);
	printf("Time for %d iterations was %g seconds\n",iter,ttot);
	printf("Each iteration took %g seconds\n",titer);

	//output results
	//writedatafiles(psi,m,n, scalefactor);
	//writeplotfile(m,n,scalefactor);

	//free un-needed arrays
	free(psi);
	free(psitmp);

	cudaFree(d_psi);
	cudaFree(d_psitmp);
	printf("... finished\n");

	return 0;
}
