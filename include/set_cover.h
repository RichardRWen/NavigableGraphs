#pragma once

#include <iostream>
#include <utility>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <utils/point_range.h>

#include "distance_matrix.h"

template <typename val_t>
class SetCoverAdjlists {
public:
    using Point_t = parlayANN::Euclidian_Point<val_t>;
    using PointRange_t = parlayANN::PointRange<Point_t>;

    PointRange_t &points;
    DistanceMatrix<val_t> dist_mat;
    SortedDistances sorted_dists;
    parlay::sequence<uint32_t> sorted_ranks;

    SetCoverAdjlists(PointRange_t &points) : points(points), dist_mat(points), sorted_dists(dist_mat) {
        sorted_ranks = parlay::sequence<uint32_t>::uninitialized(sorted_dists.indices.size());
        parlay::parallel_for(0, points.size(), [&](size_t i) {
            uint32_t *dists = sorted_dists.indices.begin() + i * points.size();
            uint32_t *ranks = sorted_ranks.begin() + i * points.size();
            for (size_t j = 0; j < points.size(); j++) {
                ranks[dists[j]] = j;
            }
        }, 1);
    }

    parlay::sequence<uint32_t> adjlist_greedy(uint32_t p) {
        parlay::sequence<uint32_t> adjlist;
        std::vector<bool> covered(points.size(), false);
        size_t uncovered_count = points.size();

        // Compute, for each point q, the set of points r it helps cover
        // To do this, search for the p in the sorted list of distances from q
        // All points before p are covered by q
        parlay::sequence<std::pair<uint32_t, uint32_t>> sets; // First element is size, second is index of first uncovered point in sorted_dists
        sets.reserve(points.size());
        for (int i = 0; i < points.size(); i++) {
            val_t *distances = dist_mat.distances.begin() + i * points.size();
            uint32_t *indices = sorted_dists.indices.begin() + i * points.size();
            uint32_t set_boundary = std::lower_bound(indices, indices + points.size(), p, [&](uint32_t a, uint32_t b) {
                return distances[a] < distances[b];
            }) - indices;
            sets.push_back({set_boundary, set_boundary});
        }
        
        // Compute a greedy set cover for a logn approximation
        // While there are uncovered points, pick the set with the most uncovered points
        // For each of its uncovered points, decrement the size of all sets that cover it
        while (uncovered_count > 0) {
            auto largest_set = std::max_element(sets.begin(), sets.end(), [](const auto &a, const auto &b) {
                return a.first < b.first;
            });
            if (largest_set->first <= 0) {
                std::cerr << "Error: Unable to cover all points." << std::endl;
                abort();
            }
            adjlist.push_back(largest_set - sets.begin());
            uint32_t *indices = sorted_dists.indices.begin() + (largest_set - sets.begin()) * points.size();
            uint32_t *ranks = sorted_ranks.begin() + (largest_set - sets.begin()) * points.size();
            for (size_t j = 0; j < largest_set->second; j++) {
                if (!covered[indices[j]]) {
                    covered[indices[j]] = true;
                    uncovered_count--;
                    for (auto &set : sets) {
                        if (ranks[indices[j]] < set.second) {
                            set.first--;
                        }
                    }
                }
            }
        }

        return adjlist;
    }

    parlay::sequence<parlay::sequence<uint32_t>> adjlists_greedy() {
        auto adjlists = parlay::sequence<parlay::sequence<uint32_t>>::uninitialized(points.size());
        parlay::parallel_for(0, points.size(), [&](size_t i) {
            adjlists[i] = adjlist_greedy(i);
        }, 1);
        return adjlists;
    }
};