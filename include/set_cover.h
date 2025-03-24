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
        covered[v] = true;
        size_t total_uncovered = points.size() - 1;

        // Compute, for each point v, the set of points u it helps cover
        // To do this, search for v in the sorted list of distances from u
        // All points before v are covered by u
        std::vector<std::vector<uint32_t>> sets(points.size()); // First element is size, second is index of first uncovered point in sorted_dists
        std::vector<uint32_t> set_boundaries;
        set_boundaries.reserve(points.size());
        for (uint32_t j = 0; j < points.size(); j++) {
            val_t *distances = dist_mat.distances(j);
            uint32_t *indices = sorted_dists.indices.begin() + j * points.size();
            uint32_t set_boundary = std::lower_bound(indices, indices + points.size(), v, [&](uint32_t a, uint32_t b) {
                return distances[a] < distances[b];
            }) - indices;
            set_boundaries.push_back(set_boundary);
            for (uint32_t i = 0; i < set_boundary; i++) {
                sets[indices[i]].push_back(j);
            }
        }
        std::vector<uint32_t> num_uncovered;
        num_uncovered.reserve(points.size());
        for (size_t i = 0; i < points.size(); i++) {
            num_uncovered.push_back(sets[i].size());
        }
        
        // Compute a greedy set cover for a logn approximation
        // While there are uncovered points, pick the set with the most uncovered points
        // For each of its uncovered points, decrement the size of all sets that cover it
        while (total_uncovered > 0) {
            auto best_set = std::max_element(num_uncovered.begin(), num_uncovered.end());
            if (*best_set == 0) {
                std::cerr << "Error: Unable to cover all points." << std::endl;
                abort();
            }
            uint32_t set_index = best_set - num_uncovered.begin();
            adjlist.push_back(set_index);
            if (*best_set == total_uncovered) break;
            total_uncovered -= *best_set;
            for (uint32_t j : sets[set_index]) {
                if (!covered[j]) {
                    covered[j] = true;
                    uint32_t *indices = sorted_dists.indices.begin() + j * points.size();
                    for (uint32_t i = 0; i < set_boundaries[j]; i++) {
                        num_uncovered[indices[i]]--;
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