#pragma once

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
    inline bool covers(size_t i, size_t s, size_t p, RankMatrix<index_t> &ranks) {
        // Check if set s covers point p in set cover instance i
        return ranks[p][s] < ranks[p][i];
    }
    template <typename index_t>
    inline auto sets_of(size_t i, size_t p, PermutationMatrix<index_t> &permutations, RankMatrix<index_t> &ranks) {
        // Get the sets that cover point p in set cover instance i
        return parlay::make_slice(permutations[p].begin(), permutations[p].begin() + ranks[p][i]);
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
                    for (; j >= 0; j--) {
                        sets[j].erase(p);
                    }
                    for (index_t v : voters[s]) {
                        auto v_sets = sets_of(i, v, permutations, ranks);
                        for (size_t k = 0; k < v_sets.size(); k++) {
                            index_t v_s = v_sets[k];
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
        std::uniform_int_distribution<index_t> dis(0, num_points - 1);
        parlay::parallel_for(0, num_points, [&](size_t i) {
            auto rnd = gen[i];
            std::unordered_set<index_t> chosen;
            chosen.reserve(est_avg_deg);
            for (int i = 0; i < est_avg_deg; i++) {
                index_t j = dis(rnd);
                if (j == i) continue;
                if (chosen.find(j) != chosen.end()) continue;
                chosen.insert(j);
                adjlists[i].push_back(j);
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
        size_t block_size = num_points / 2 / parlay::internal::num_workers();
        size_t num_blocks = (num_points + block_size - 1) / block_size;
        parlay::parallel_for(0, num_blocks, [&](size_t b) {
            size_t start = b * block_size;
            size_t end = std::min(start + block_size, num_points);
            for (size_t i = start; i < end; i++) {
                if (tot_deg > est_tot_deg) break;
                auto rnd = gen[i];
                std::shuffle(uncovered[i].begin(), uncovered[i].end(), rnd);
                minimum_adjacency_list(num_points, i, uncovered[i], adjlists[i], permutations, ranks);
                tot_deg += adjlists[i].size();
            }
        }, 1);

        if (tot_deg > est_tot_deg) return {false, {}};
        return {true, adjlists};
    }

    template <typename index_t, typename value_t, typename PointSet>
    std::vector<std::vector<index_t>> minimum_navigable_graph(PointSet &points) {
        // Compute the distance, permutation, and rank matrices
        DistanceMatrix<value_t> distances(points);
        PermutationMatrix permutations(distances);
        RankMatrix ranks(permutations);

        // Exponential search for the optimal number of edges
        size_t avg_deg = points.size();
        while (true) {
            auto [success, adjlists] = minimum_navigable_graph_opt<index_t>(points.size(), avg_deg, permutations, ranks);
            if (success) return adjlists;
            avg_deg *= 2;
        }
    }
};