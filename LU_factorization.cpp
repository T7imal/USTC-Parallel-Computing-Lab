#include <iostream>
#include <omp.h>
#include <cstdio>
#include <chrono>

int A[1000][1000];
// int L[1000][1000];
// int U[1000][1000];

void LU_factorization()
{
    int n;
    std::cin >> n;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            scanf("%d", &A[i][j]);
        }
    }

#pragma omp parallel
    {
        for (int k = 0; k < n; k++)
        {
#pragma omp for
            for (int i = k + 1; i < n; i++)
            {
                A[i][k] = A[i][k] / A[k][k];
                for (int j = k + 1; j < n; j++)
                {
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                }
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i < j)
            {
                printf("%d ", 0);
            }
            else if (i == j)
            {
                printf("%d ", 1);
            }
            else
            {
                printf("%d ", A[i][j]);
            }
        }
        printf("\n");
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i <= j)
            {
                printf("%d ", A[i][j]);
            }
            else
            {
                printf("%d ", 0);
            }
        }
        printf("\n");
    }
}

int main()
{
    omp_set_num_threads(8);
    auto start = omp_get_wtime();
    // LU_factorization();
    LU_factorization();
    auto end = omp_get_wtime();
    std::cout << "Time: " << end - start << std::endl;

    return 0;
}
