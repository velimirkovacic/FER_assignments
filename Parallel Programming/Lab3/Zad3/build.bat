@echo off
set NVCC_FLAGS=-m64
set GPP_FLAGS=-m64

nvcc -c cfd.cu -o cfd.o -arch=sm_60
nvcc -c arraymalloc.cu -o arraymalloc.o
nvcc -c jacobi.cu -o jacobi.o
nvcc -c boundary.cu -o boundary.o
nvcc -c cfdio.cu -o cfdio.o

nvcc -o program cfd.o cfdio.o boundary.o jacobi.o arraymalloc.o -arch=sm_60


del /f *.o
