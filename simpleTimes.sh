# Script used to average the execution times for each processor count. The resultant times are outputed that text files that were used to make the plot

mpicc -std=gnu99 simpleTimes.c -o simpleTimes -lm
for i in {0..25}
do
	mpirun -np 1 ./simpleTimes -o simpleTimes >> 1x1SimpleTimes.txt
done
for i in {0..25}
do
	mpirun -np 4 ./simpleTimes -o simpleTimes >> 2x2SimpleTimes.txt
done
for i in {0..25}
do
	mpirun -np 16 ./simpleTimes -o simpleTimes >> 4x4SimpleTimes.txt
done
for i in {0..25}
do
	mpirun -np 64 ./simpleTimes -o simpleTimes >> 8x8SimpleTimes.txt
done
for i in {0..25}
do
	mpirun -np 256 ./simpleTimes -o simpleTimes >> 16x16SimpleTimes.txt
done
