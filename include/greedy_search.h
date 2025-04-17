#pragma once

#include <cstdint>
#include <vector>
#include <utility>

#include <parlay/sequence.h>

#include "point_set.h"

template <typename Graph, typename value_t>
std::pair<uint32_t, uint32_t> greedy_search(Graph &graph, PointSet<value_t> &points, uint32_t source, uint32_t query) {
    parlay::sequence<bool> visited(points.size(), false);
    uint32_t current = source;
    value_t current_dist = points[source].distance(points[query]);
    uint32_t dist_comps = 1;

    while (!visited[current]) {
        visited[current] = true;
        for (uint32_t neighbor : graph[current]) {
            if (visited[neighbor]) continue;
            value_t dist = points[neighbor].distance(points[query]);
            dist_comps++;
            if (dist < current_dist) {
                if (dist == 0) {
                    return std::make_pair(neighbor, dist_comps);
                }
                visited[current] = true;
                current = neighbor;
                current_dist = dist;
            }
            else {
                visited[neighbor] = true;
            }
        }
    }
    return std::make_pair(current, dist_comps);
}