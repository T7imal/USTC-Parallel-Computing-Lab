#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double random_double() { return (double)rand() / (double)RAND_MAX; }

int main()
{
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    srand(time(NULL));

    int num = 1000000;
    double a = 1.0, b = 1.0;
    double l = 1.0;

    int hits = 0;
    for (int i = 0; i < num; i++)
    {
        double mid_x = random_double() * a;
        double mid_y = random_double() * b;
        double angle = random_double() * 2 * M_PI;

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

    double probability = (double)hits / (double)num;
    double pi_estimate = (2.0 * l * (a + b) - l * l) / (a * b * probability);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e6;
    printf("Time taken: %f ms\n", time_taken);

    printf("Estimate for pi: %f\n", pi_estimate);

    return 0;
}