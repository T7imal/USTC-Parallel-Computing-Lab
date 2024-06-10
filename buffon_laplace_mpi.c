#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();

    int num = 10000000 / size;
    double a = 1.0, b = 1.0;
    double l = 1.0;

    int hits = 0;
    unsigned int seed = time(NULL) + rank;
    for (int i = 0; i < num; i++)
    {
        double mid_x = (double)rand_r(&seed) / (double)RAND_MAX * a;
        double mid_y = (double)rand_r(&seed) / (double)RAND_MAX * b;
        double angle = (double)rand_r(&seed) / (double)RAND_MAX * 2 * M_PI;

        double needle_tip_x = mid_x + (l / 2.0) * cos(angle);
        double needle_tip_y = mid_y + (l / 2.0) * sin(angle);
        double needle_tail_x = mid_x - (l / 2.0) * cos(angle);
        double needle_tail_y = mid_y - (l / 2.0) * sin(angle);

        if (needle_tip_x < 0 || needle_tip_x > a || needle_tip_y < 0 ||
            needle_tip_y > b || needle_tail_x < 0 || needle_tail_x > a ||
            needle_tail_y < 0 || needle_tail_y > b)
        {
            hits++;
        }
    }

    int total_hits;
    MPI_Reduce(&hits, &total_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        double probability = (double)total_hits / (double)(num * size);
        double pi_estimate =
            (2.0 * l * (a + b) - l * l) / (a * b * probability);
        printf("Estimate for pi: %f\n", pi_estimate);
        double time_taken = (end - start) * 1e3;
        printf("Time taken: %f ms\n", time_taken);
    }

    MPI_Finalize();

    return 0;
}