#pragma once

#include <cstdint>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/parallel.h>

#include "point_set.h"

template <typename val_t>
class DistanceMatrix {
    size_t _size;
    parlay::sequence<val_t> dists;

public:
    template <typename Points>
    DistanceMatrix(Points &points) : _size(points.size()) {
        dists = parlay::sequence<val_t>::uninitialized(_size * _size);

        parlay::parallel_for(0, _size, [&](size_t i) {
            dists[i * _size + i] = 0;
            for (size_t j = i + 1; j < _size; j++) {
                val_t dist = points[i].distance(points[j]);
                dists[i * _size + j] = dist;
                dists[j * _size + i] = dist;
            }
        }, 1);
    }

    inline size_t size() const {
        return _size;
    }

    inline val_t *operator[](size_t i) {
        return dists.begin() + i * _size;
    }
};

class PermutationMatrix {
    size_t _size;
    parlay::sequence<uint32_t> indices;

public:
    template <typename val_t>
    PermutationMatrix(DistanceMatrix<val_t> &dist_mat) : _size(dist_mat.size()) {
        indices = parlay::sequence<uint32_t>::uninitialized(_size * _size);
        parlay::parallel_for(0, _size, [&](size_t i) {
            for (size_t j = 0; j < _size; j++) {
                indices[i * _size + j] = j;
            }
        }, 1);
        parlay::parallel_for(0, _size, [&](size_t i) {
            val_t *distances = dist_mat[i];
            std::sort(indices.begin() + i * _size, indices.begin() + (i + 1) * _size, [&](uint32_t a, uint32_t b) {
                return distances[a] < distances[b];
            });
        }, 1);
    }

    inline size_t size() const {
        return _size;
    }

    inline uint32_t *operator[](size_t i) {
        return indices.begin() + i * _size;
    }
};

class RankMatrix {
    size_t _size;
    parlay::sequence<uint32_t> ranks;

public:
    RankMatrix(PermutationMatrix &perm_mat) : _size(perm_mat.size()) {
        ranks = parlay::sequence<uint32_t>::uninitialized(_size * _size);
        parlay::parallel_for(0, _size, [&](size_t i) {
            uint32_t *indices = perm_mat[i];
            for (size_t j = 0; j < _size; j++) {
                ranks[i * _size + indices[j]] = j;
            }
        }, 1);
    }

    inline size_t size() const {
        return _size;
    }

    inline uint32_t *operator[](size_t i) {
        return ranks.begin() + i * _size;
    }
};