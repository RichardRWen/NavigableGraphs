#pragma once

#include <iostream>
#include <utility>
#include <mutex>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <utils/point_range.h>

#include "point_set.h"
#include "distance_matrix.h"

template <typename val_t>
class SetCoverAdjlists {
public:
    PointSet<val_t> &points;
    DistanceMatrix<val_t> dist_mat;
    SortedDistances sorted_dists;
    parlay::sequence<uint32_t> sorted_ranks;

    SetCoverAdjlists(PointSet<val_t> &points) : points(points), dist_mat(points), sorted_dists(dist_mat) {
        sorted_ranks = parlay::sequence<uint32_t>::uninitialized(sorted_dists.indices.size());
        parlay::parallel_for(0, points.size(), [&](size_t i) {
            uint32_t *dists = sorted_dists.indices.begin() + i * points.size();
            uint32_t *ranks = sorted_ranks.begin() + i * points.size();
            for (size_t j = 0; j < points.size(); j++) {
                ranks[dists[j]] = j;
            }
        }, 1);
    }

    // std::mutex cout_lock;
    std::vector<uint32_t> adjlist_greedy(uint32_t v) {
        // cout_lock.lock();
        // std::cout << v << std::endl;
        // cout_lock.unlock();

        std::vector<uint32_t> adjlist;
        std::vector<bool> covered(points.size(), false);
        size_t uncovered_count = points.size();

        // Compute, for each point v, the set of points u it helps cover
        // To do this, search for v in the sorted list of distances from u
        // All points before v are covered by u
        std::vector<std::pair<uint32_t, uint32_t>> sets; // First element is size, second is index of first uncovered point in sorted_dists
        sets.reserve(points.size());
        for (int i = 0; i < points.size(); i++) {
            val_t *distances = dist_mat.distances.begin() + i * points.size();
            uint32_t *indices = sorted_dists.indices.begin() + i * points.size();
            uint32_t set_boundary = std::upper_bound(indices, indices + points.size(), v, [&](uint32_t a, uint32_t b) {
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
                    for (size_t k = 0; k < points.size(); k++) {
                        if (sorted_ranks[k * points.size() + indices[j]] < sets[k].second) {
                            sets[k].first--;
                        }
                    }
                }
            }
        }

        return adjlist;
    }

    parlay::sequence<std::vector<uint32_t>> adjlists_greedy() {
        /*auto adjlists = parlay::sequence<std::vector<uint32_t>>::uninitialized(points.size());
        parlay::parallel_for(0, points.size(), [&](size_t i) {
            adjlists[i] = adjlist_greedy(i);
        }, 1);*/
        auto adjlists = parlay::tabulate(points.size(), [&](size_t i) {
            return adjlist_greedy(i);
        });
        return adjlists;
    }
};