#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

int main()
{
    omp_set_num_threads(8);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    int num = 10000000;
    double a = 1.0, b = 1.0;
    double l = 1.0;

    int hits = 0;
#pragma omp parallel reduction(+ : hits)
    {
        unsigned int seed = time(NULL);
#pragma omp for
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
    }

    double probability = (double)hits / (double)num;
    double pi_estimate = (2.0 * l * (a + b) - l * l) / (a * b * probability);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken =
        (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
    printf("Time taken: %f ms\n", time_taken);

    printf("Estimate for pi: %f\n", pi_estimate);

    return 0;
}