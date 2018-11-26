# Script used to average the execution times for each processor count. The resultant times are outputed that text files that were used to make the plot

mpicc -std=gnu99 cannonTimes.c -o cannonTimes -lm
for i in {0..25}
do
	mpirun -np 1 ./cannonTimes -o cannonTimes >> 1x1CannonTimes.txt
done
for i in {0..25}
do
	mpirun -np 4 ./cannonTimes -o cannonTimes >> 2x2CannonTimes.txt
done
for i in {0..25}
do
	mpirun -np 16 ./cannonTimes -o cannonTimes >> 4x4CannonTimes.txt
done
for i in {0..25}
do
	mpirun -np 64 ./cannonTimes -o cannonTimes >> 8x8CannonTimes.txt
done
for i in {0..25}
do
	mpirun -np 256 ./cannonTimes -o cannonTimes >> 16x16CannonTimes.txt
done
