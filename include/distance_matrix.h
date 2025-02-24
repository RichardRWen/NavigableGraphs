#pragma once

#include <cstdint>
#include <algorithm>

#include <parlay/parallel.h>

#include "point_set.h"

template <typename val_t>
class DistanceMatrix {
public:
    size_t size;
    parlay::sequence<val_t> distances;

    DistanceMatrix(PointSet<val_t> &points) : size(points.size()) {
        distances = parlay::sequence<val_t>::uninitialized(points.size() * points.size());

        parlay::parallel_for(0, points.size(), [&](size_t i) {
            distances[i * points.size() + i] = 0;
            for (size_t j = i + 1; j < points.size(); j++) {
                val_t dist = points[i].distance(points[j]);
                distances[i * points.size() + j] = dist;
                distances[j * points.size() + i] = dist;
            }
        }, 1);
    }
};

class SortedDistances {
public:
    parlay::sequence<uint32_t> indices;

    template <typename val_t>
    SortedDistances(DistanceMatrix<val_t> &dist_mat) {
        indices = parlay::sequence<uint32_t>::uninitialized(dist_mat.size * dist_mat.size);
        parlay::parallel_for(0, dist_mat.size, [&](size_t i) {
            for (size_t j = 0; j < dist_mat.size; j++) {
                indices[i * dist_mat.size + j] = j;
            }
        }, 1);
        parlay::parallel_for(0, dist_mat.size, [&](size_t i) {
            val_t *distances = dist_mat.distances.begin() + i * dist_mat.size;
            std::sort(indices.begin() + i * dist_mat.size, indices.begin() + (i + 1) * dist_mat.size, [&](uint32_t a, uint32_t b) {
                return distances[a] < distances[b];
            });
        }, 1);
    }
};