#pragma once

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <utility>
#include <vector>
#include <unordered_set>
#include <atomic>
#include <mutex>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/slice.h>
#include <parlay/random.h>

#include "point_set.h"
#include "distance_matrix.h"

namespace MNG {
    template <typename index_t>
    inline bool covers(index_t i, index_t s, index_t p, const RankMatrix<index_t> &ranks) {
        // Check if set s covers point p in set cover instance i
        return ranks[p][s] < ranks[p][i];
    }
    template <typename index_t>
    inline auto sets_of(index_t i, index_t p, const PermutationMatrix<index_t> &permutations, const RankMatrix<index_t> &ranks) {
        // Get the sets that cover point p in set cover instance i
        return parlay::make_slice(permutations[p], permutations[p] + ranks[p][i]);
    }

    template <typename index_t>
    void minimum_adjacency_list(size_t n, index_t i, std::vector<index_t> &uncovered, std::vector<index_t> &adjlist, const PermutationMatrix<index_t> &permutations, const RankMatrix<index_t> &ranks) {
        size_t logn = std::ceil(std::log2(n));
        std::vector<std::unordered_set<index_t>> voters(n);
        for (size_t j = 0; j < voters.size(); j++) {
            voters[j].reserve(logn);
        }

        while (!uncovered.empty()) {
            // Find an uncovered point
            index_t p = uncovered.back();
            uncovered.pop_back();
            bool is_covered = false;
            for (auto s : adjlist) {
                if (covers(i, s, p, ranks)) {
                    is_covered = true;
                    break;
                }
            }
            if (is_covered) continue;

            // Vote for the sets that cover p
            auto sets = sets_of(i, p, permutations, ranks);
            for (size_t j = 0; j < sets.size(); j++) {
                index_t s = sets[j];
                if (voters[s].size() >= logn - 1) {
                    // If the set has enough votes, add it to the adjacency list and remove its voters
                    adjlist.push_back(s);
                    for (; j > 0; j--) {
                        voters[sets[j - 1]].erase(p);
                    }
                    for (index_t v : voters[s]) {
                        auto v_sets = sets_of(i, v, permutations, ranks);
                        for (index_t v_s : v_sets) {
                            if (v_s != s) {
                                voters[v_s].erase(v);
                            }
                        }
                    }
                    voters[s].clear();
                    break;
                }
                else voters[s].insert(p);
            }
        }
    }

    template <typename index_t>
    std::pair<bool, std::vector<std::vector<index_t>>> minimum_navigable_graph_opt(size_t num_points, size_t opt_deg, const PermutationMatrix<index_t> &permutations, const RankMatrix<index_t> &ranks) {
        std::vector<std::vector<index_t>> adjlists(num_points);
        size_t est_avg_deg = opt_deg * std::ceil(std::log2(num_points)); // Assuming num_points > 1
        size_t est_tot_deg = 2 * est_avg_deg * num_points;

        // Add random edges to each adjacency list
        parlay::random_generator gen(0);
        std::uniform_int_distribution<index_t> dis(0, num_points - 2);
        parlay::parallel_for(0, num_points, [&](size_t i) {
            auto rnd = gen[i];
            std::unordered_set<index_t> chosen;
            chosen.reserve(est_avg_deg);
            for (int j = 0; j < est_avg_deg; j++) {
                index_t k = dis(rnd);
                if (k >= i) k++;
                if (chosen.find(k) != chosen.end()) continue;
                chosen.insert(k);
                adjlists[i].push_back(k);
            }
        }, 1);

        // Initialize sets of uncovered points
        std::vector<std::vector<index_t>> uncovered(num_points);
        std::vector<std::mutex> locks(num_points);
        size_t uncovered_per_instance = num_points / opt_deg;
        parlay::parallel_for(0, num_points, [&](size_t i) {
            for (size_t j = 1; j < uncovered_per_instance; j++) {
                index_t p = permutations[i][j];
                std::lock_guard<std::mutex> lock(locks[p]);
                uncovered[p].push_back(i);
            }
        }, 1);

        // Compute adjacency lists using set cover
        std::atomic<size_t> tot_deg = 0;
        size_t block_size = num_points / 2 / parlay::num_workers();
        if (block_size < 1) block_size = 1;
        size_t num_blocks = (num_points + block_size - 1) / block_size;
        // parlay::parallel_for(0, num_blocks, [&](size_t b) {
        for (size_t b = 0; b < num_blocks; b++) {
            size_t start = b * block_size;
            size_t end = std::min(start + block_size, num_points);
            for (index_t i = start; i < end; i++) {
                if (tot_deg > est_tot_deg) break;
                auto rnd = gen[i];
                std::shuffle(uncovered[i].begin(), uncovered[i].end(), rnd);
                minimum_adjacency_list<index_t>(num_points, i, uncovered[i], adjlists[i], permutations, ranks);
                tot_deg += adjlists[i].size();
            }
        // }, 1);
        }

        if (tot_deg > est_tot_deg) return {false, {}};
        return {true, adjlists};
    }

    template <typename index_t, typename value_t, typename PointSet>
    std::vector<std::vector<index_t>> minimum_navigable_graph(PointSet &points) {
        // Compute the distance, permutation, and rank matrices
        DistanceMatrix<value_t> distances(points);
        PermutationMatrix<index_t> permutations(distances);
        RankMatrix<index_t> ranks(distances, permutations);

        // Exponential search for the optimal number of edges
        size_t avg_deg = 1;
        while (true) {
            auto [success, adjlists] = minimum_navigable_graph_opt<index_t>(points.size(), avg_deg, permutations, ranks);
            if (success) return adjlists;
            avg_deg *= 2;
        }
    }
};