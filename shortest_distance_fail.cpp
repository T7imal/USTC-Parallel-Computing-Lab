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

std::vector<int> dist;
std::vector<std::set<int>> buckets;
// std::vector<std::list<Edge>> graph;
std::vector<std::vector<Edge>> light_edges;
std::vector<std::vector<Edge>> heavy_edges;
int delta = 100;
int min_bucket = 0;
int max_bucket = 0;

void relax(int u, int d)
{
    if (d < dist[u])
    {
#pragma omp critical
        {
            // 双重检查
            if (d < dist[u])
            {
                int old_bucket = dist[u] == INF ? INF : dist[u] / delta;
                int new_bucket = d / delta;
                dist[u] = d;
                if (old_bucket != INF)
                {
                    buckets[old_bucket].erase(u);
                }
                max_bucket = std::max(max_bucket, new_bucket);
                if (max_bucket >= buckets.size())
                {
                    buckets.resize(max_bucket + 1);
                }
                buckets[new_bucket].insert(u);
            }
        }
    }
}

void deltaStepping(int src)
{
    buckets.resize(1);
    relax(src, 0);
    while (min_bucket <= max_bucket)
    {
        // printf("deal with bucket %d\n", min_bucket);
        std::set<int> used_vertices;
        std::vector<int> tmpVec;
#pragma omp parallel
        {
            while (!buckets[min_bucket].empty())
            {
#pragma omp barrier
#pragma omp single
                {
                    tmpVec.clear();
                    tmpVec.insert(tmpVec.begin(), buckets[min_bucket].begin(), buckets[min_bucket].end());
                    std::set_union(used_vertices.begin(), used_vertices.end(), buckets[min_bucket].begin(),
                                   buckets[min_bucket].end(), std::inserter(used_vertices, used_vertices.begin()));
                    buckets[min_bucket].clear();
                }
#pragma omp for
                for (int i = 0; i < tmpVec.size(); i++)
                {
                    int u = tmpVec[i];
                    for (auto &edge : light_edges[u])
                    {
                        int v = edge.to;
                        int w = edge.weight;
                        // relax light edges
                        relax(v, dist[u] + w);
                    }
                }
            }

            std::vector<int> used_vertices_vec(used_vertices.begin(), used_vertices.end());
#pragma omp for nowait
            for (int i = 0; i < used_vertices.size(); ++i)
            {
                int u = used_vertices_vec[i];
                for (auto &edge : heavy_edges[u])
                {
                    int v = edge.to;
                    int w = edge.weight;
                    // relax heavy edges
                    relax(v, dist[u] + w);
                }
            }
        }
        min_bucket++;
    }
}

int main()
{
    omp_set_num_threads(8);

    int V, E, src;
    scanf("%d %d %d", &V, &E, &src);

    // graph.resize(V);
    light_edges.resize(V);
    heavy_edges.resize(V);
    dist.resize(V, INF);

    int min_weight = INF;

    for (int i = 0; i < E; ++i)
    {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        // graph[u].push_back({v, w});
        // graph[v].push_back({u, w});
        if (w <= delta)
        {
            light_edges[u].push_back({v, w});
            light_edges[v].push_back({u, w});
        }
        else
        {
            heavy_edges[u].push_back({v, w});
            heavy_edges[v].push_back({u, w});
        }
    }

    auto start = omp_get_wtime();

    deltaStepping(src);

    auto end = omp_get_wtime();

    printf("Time: %lf\n", end - start);

    // for (int i = 0; i < V; ++i)
    // {
    //     if (dist[i] == INF)
    //         printf("INF ");
    //     else
    //         printf("%d ", dist[i]);
    // }

    return 0;
}