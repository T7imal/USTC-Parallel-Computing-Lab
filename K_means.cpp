#include <mpi.h>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <chrono>
#include <random>

void K_means()
{
    MPI_Init(NULL, NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // int n, m, k;
    int args[3];
    if (rank == 0)
    {
        // scanf("%d %d %d", &n, &m, &k);
        scanf("%d %d %d", args, args + 1, args + 2);
    }
    // MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(args, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int n = args[0];
    int m = args[1];
    int k = args[2];

    double datas[n][m];
    double centers[k][m];
    if (rank == 0)
    {
        // 记录输入数据的上下界
        double upper_bound[m];
        double lower_bound[m];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                scanf("%lf", &datas[i][j]);
                if (i == 0)
                {
                    upper_bound[j] = datas[i][j];
                    lower_bound[j] = datas[i][j];
                }
                else
                {
                    if (datas[i][j] > upper_bound[j])
                    {
                        upper_bound[j] = datas[i][j];
                    }
                    if (datas[i][j] < lower_bound[j])
                    {
                        lower_bound[j] = datas[i][j];
                    }
                }
            }
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        // 随机生成中心点
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < m; j++)
            {
                std::uniform_real_distribution<double> dis(lower_bound[j], upper_bound[j]);
                centers[i][j] = dis(gen);
            }
        }
    }

    // 分配数据点到不同的进程
    int n_per_proc = std::ceil((double)n / size);
    double local_datas[n_per_proc][m];
    if (rank == 0)
    {
        // 在主进程中，将数据点分配到其他进程

        int sendcounts[size];
        int displs[size];
        for (int i = 0; i < size - 1; i++)
        {
            sendcounts[i] = n_per_proc * m;
            displs[i] = i * n_per_proc * m;
        }
        sendcounts[size - 1] = (n - (size - 1) * n_per_proc) * m;
        displs[size - 1] = (size - 1) * n_per_proc * m;
        MPI_Scatterv(datas, sendcounts, displs, MPI_DOUBLE, local_datas, n_per_proc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        if (rank == size - 1)
        {
            n_per_proc = n - (size - 1) * n_per_proc;
        }
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, local_datas, n_per_proc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(centers, k * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int it = 0; it < 100; it++)
    {
        int closest_centers[n_per_proc];
        double min_distances[n_per_proc];
        for (int i = 0; i < n_per_proc; i++)
        {
            closest_centers[i] = -1;
            min_distances[i] = 1e9;
        }

        // 计算每个进程的数据点到中心点的距离
        for (int i = 0; i < n_per_proc; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double distance_square = 0;
                for (int dim = 0; dim < m; dim++)
                {
                    double diff = local_datas[i][dim] - centers[j][dim];
                    distance_square += diff * diff;
                }
                if (distance_square < min_distances[i])
                {
                    min_distances[i] = distance_square;
                    closest_centers[i] = j;
                }
            }
        }

        double new_centers[k][m];
        int cluster_sizes[k];
        for (int i = 0; i < k; i++)
        {
            cluster_sizes[i] = 0;
            for (int j = 0; j < m; j++)
            {
                new_centers[i][j] = 0;
            }
        }
        for (int i = 0; i < n_per_proc; i++)
        {
            cluster_sizes[closest_centers[i]]++;
            for (int dim = 0; dim < m; dim++)
            {
                new_centers[closest_centers[i]][dim] += local_datas[i][dim];
            }
        }

        // 汇总每个进程的数据
        double new_centers_sum[k][m];
        int cluster_sizes_sum[k];
        MPI_Allreduce(new_centers, new_centers_sum, k * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(cluster_sizes, cluster_sizes_sum, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < m; j++)
            {
                centers[i][j] = new_centers_sum[i][j] / cluster_sizes_sum[i];
            }
        }
    }

    double total_distance = 0;
    for (int i = 0; i < n_per_proc; i++)
    {
        double min_distance = 1e9;
        for (int j = 0; j < k; j++)
        {
            double distance = 0;
            for (int dim = 0; dim < m; dim++)
            {
                double diff = local_datas[i][dim] - centers[j][dim];
                distance += diff * diff;
            }
            if (distance < min_distance)
            {
                min_distance = distance;
            }
        }
        total_distance += std::sqrt(min_distance);
    }

    double total_distance_sum;
    MPI_Reduce(&total_distance, &total_distance_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("%.2f\n", total_distance_sum);
    }

    MPI_Finalize();
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    K_means();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Program executed in %.2f seconds\n", diff.count());

    return 0;
}