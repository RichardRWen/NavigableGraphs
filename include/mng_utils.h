#pragma once

#include <cstdint>
#include <algorithm>

#include <parlay/sequence.h>
#include <parlay/parallel.h>

#include "point_set.h"

template <typename value_t>
class DistanceMatrix {
    size_t _size;
    parlay::sequence<value_t> dists;

public:
    template <typename Points>
    DistanceMatrix(Points &points) : _size(points.size()) {
        dists = parlay::sequence<value_t>::uninitialized(_size * _size);
        auto &matrix = *this;
        parlay::parallel_for(0, _size, [&](size_t i) {
            matrix[i][i] = 0;
            for (size_t j = i + 1; j < _size; j++) {
                value_t dist = points[i].distance(points[j]);
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }, 1);
    }

    inline size_t size() const {
        return _size;
    }

    inline value_t *operator[](size_t i) {
        return dists.begin() + i * _size;
    }
    inline const value_t *operator[](size_t i) const {
        return dists.begin() + i * _size;
    }
};

template <typename index_t = uint32_t>
class PermutationMatrix {
    size_t _size;
    parlay::sequence<index_t> indices;

public:
    template <typename value_t>
    PermutationMatrix(DistanceMatrix<value_t> &dist_mat) : _size(dist_mat.size()) {
        indices = parlay::sequence<uint32_t>::uninitialized(_size * _size);
        auto &matrix = *this;
        parlay::parallel_for(0, _size, [&](size_t i) {
            for (size_t j = 0; j < _size; j++) {
                matrix[i][j] = j;
            }
        }, 1);
        parlay::parallel_for(0, _size, [&](size_t i) {
            value_t *distances = dist_mat[i];
            std::sort(matrix[i], matrix[i + 1], [&](uint32_t a, uint32_t b) {
                return distances[a] < distances[b];
            });
        }, 1);
    }

    inline size_t size() const {
        return _size;
    }

    inline index_t *operator[](size_t i) {
        return indices.begin() + i * _size;
    }
    inline const index_t *operator[](size_t i) const {
        return indices.begin() + i * _size;
    }
};

template <typename index_t = uint32_t>
class RankMatrix {
    size_t _size;
    parlay::sequence<index_t> ranks;

public:
    template <typename value_t>
    RankMatrix(DistanceMatrix<value_t> &dist_mat, PermutationMatrix<index_t> &perm_mat) : _size(dist_mat.size()) {
        ranks = parlay::sequence<uint32_t>::uninitialized(_size * _size);
        auto &matrix = *this;
        parlay::parallel_for(0, _size, [&](size_t i) {
            uint32_t *indices = perm_mat[i];
            for (size_t j = 0; j < _size; j++) {
                matrix[i][indices[j]] = j;
            }
            for (size_t j = 1; j < _size; j++) {
                if (dist_mat[i][j] == dist_mat[i][j - 1]) {
                    matrix[i][j] = matrix[i][j - 1];
                }
            }
        }, 1);
    }

    inline size_t size() const {
        return _size;
    }

    inline index_t *operator[](size_t i) {
        return ranks.begin() + i * _size;
    }
    inline const index_t *operator[](size_t i) const {
        return ranks.begin() + i * _size;
    }
};