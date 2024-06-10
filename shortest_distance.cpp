#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <limits>
#include <omp.h>

const int INF = std::numeric_limits<int>::max();

struct Edge
{
    int to, weight;
};

void shortest_distance()
{
    int V, E, src;
    scanf("%d %d %d", &V, &E, &src);

    std::vector<std::list<Edge>> graph(V);

    int min_weight = INF;

    for (int i = 0; i < E; ++i)
    {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        graph[u].push_back({v, w});
        graph[v].push_back({u, w});
        min_weight = std::min(min_weight, w);
    }

    std::vector<int> dist;

    int delta = 10000;

    int n = graph.size();
    dist.assign(n, INF);
    dist[src] = 0;

    std::vector<std::set<int>> buckets(n);
    int maxBucket = 0;
    int minBucket = 0;
    int tmpMaxBucket = 0;
    buckets[0].insert(src);
#pragma omp parallel
    {
        while (maxBucket >= minBucket)
        {
#pragma omp barrier
// printf("minBucket = %d, maxBucket = %d\n", minBucket, maxBucket);
#pragma omp single
            {
                tmpMaxBucket = maxBucket;
                for (int i = minBucket; i <= maxBucket; ++i)
                {
                    if (buckets[i].empty())
                    {
                        if (i == maxBucket)
                        {
                            // printf("all buckets are empty\n");
                            minBucket = maxBucket + 1;
                        }
                        continue;
                    }
                    minBucket = i;
                    // printf("set minBucket to %d\n", minBucket);
                    break;
                }
            }
#pragma omp for schedule(static)
            for (int i = minBucket; i <= maxBucket; ++i)
            {
                while (!buckets[i].empty())
                {
                    // printf("deal with bucket %d\n", i);
                    int u = -1;
#pragma omp critical
                    {
                        // 双重判断，防止多线程下的竞争
                        if (!buckets[i].empty())
                        {
                            u = *buckets[i].begin();
                            buckets[i].erase(buckets[i].begin());
                        }
                    }
                    if (u == -1)
                    {
                        continue;
                    }
                    for (const auto &edge : graph[u])
                    {
                        int v = edge.to;
                        int w = edge.weight;
                        int newDist = dist[u] + w;
                        if (newDist < dist[v])
                        {
#pragma omp critical
                            {
                                // 双重判断，防止多线程下的竞争
                                if (newDist < dist[v])
                                {
                                    if (dist[v] != INF)
                                    {
                                        int oldBucket = dist[v] / delta;
                                        buckets[oldBucket].erase(v);
                                    }
                                    dist[v] = newDist;
                                    int newBucket = newDist / delta;
                                    if (newBucket >= buckets.size())
                                    {
                                        buckets.resize(newBucket + 1);
                                    }
                                    buckets[newBucket].insert(v);
                                    tmpMaxBucket = std::max(tmpMaxBucket, newBucket);
                                }
                            }
                        }
                    }
                }
            }
#pragma omp single
            {
                if (tmpMaxBucket > maxBucket)
                {
                    maxBucket = tmpMaxBucket;
                }
                else
                {
                    for (int i = maxBucket; i >= minBucket; --i)
                    {
                        if (!buckets[i].empty())
                        {
                            maxBucket = i;
                            break;
                        }
                    }
                }
            }
        }
    }

    // for (int i = 0; i < V; ++i)
    // {
    //     if (dist[i] == INF)
    //         printf("INF ");
    //     else
    //         printf("%d ", dist[i]);
    // }
}

int main()
{
    omp_set_num_threads(8);

    auto start = omp_get_wtime();

    shortest_distance();

    auto end = omp_get_wtime();
    std::cout << "Time: " << end - start << "s" << std::endl;

    return 0;
}