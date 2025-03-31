#include <iostream>
#include <cstdint>
#include <cstring>

#include <sys/mman.h>

#include <parlay/sequence.h>
#include <parlay/internal/get_time.h>

#include <utils/types.h>
#include <utils/graph.h>
#include <utils/euclidian_point.h>
#include <utils/point_range.h>
#include <utils/parse_results.h>
#include <utils/check_nn_recall.h>

#include "point_set.h"
#include "distance_matrix.h"
#include "set_cover.h"
#include "greedy_search.h"

#define PARALLEL 1
#define MODE 1

int main(int argc, char* argv[]) {
    std::string test = "sift_10K";
    size_t sample_size = -1ULL;
    if (argc > 1) {
        test = argv[1];
    }
    if (argc > 2) {
        sample_size = std::stoul(argv[2]);
    }

    using index_t = uint32_t;
    using value_t = float;
    using GroundTruth_t = parlayANN::groundTruth<index_t>;
    using Graph_t = parlayANN::Graph<index_t>;

    // Load the points
    std::cout << "Loading test: " << test << std::endl;
    PointSet points("/ssd1/richard/navgraphs/" + test + ".fbin", sample_size);
    PointSet queries("/ssd1/richard/navgraphs/" + test + ".fbin", sample_size);
    GroundTruth_t groundtruth(("/ssd1/richard/navgraphs/" + test + ".gt").data());

    // Compute the adjacency lists
    std::cout << "Computing adjacency lists using greedy set cover" << std::endl;
    parlay::internal::timer timer;
    timer.start();
    SetCoverAdjlists<value_t> set_cover(points);

    #if MODE == 0 // Greedy
        #if PARALLEL
            auto adjlists = set_cover.adjlists_greedy();
        #else
            std::vector<std::vector<index_t>> adjlists;
            for (size_t i = 0; i < points.size(); i++) {
                std::cout << "Computing adjacency list for point " << i << std::endl;
                adjlists.push_back(set_cover.adjlist_greedy(i));
            }
        #endif
    #elif MODE == 1 // Sampling
        #if PARALLEL
            auto adjlists = set_cover.adjlists_sampling();
        #else
            parlay::random_generator gen(0);
            std::vector<std::vector<index_t>> adjlists;
            for (size_t i = 0; i < points.size(); i++) {
                std::cout << "Computing adjacency list for point " << i << std::endl;
                auto r = gen[i];
                adjlists.push_back(set_cover.adjlist_sampling(i, r));
            }
        #endif
    #else
        #error "Invalid mode"
    #endif

    std::cout << "Adjacency lists computed in " << timer.next_time() << " seconds" << std::endl;

    // Output basic statistics
    auto adjlist_sizes = parlay::map(adjlists, [](auto &adjlist) { return adjlist.size(); });
    size_t max_degree = parlay::reduce(adjlist_sizes, parlay::maxm<size_t>());
    double avg_degree = parlay::reduce(adjlist_sizes) / (double)adjlists.size();
    std::cout << "Max degree: " << max_degree << std::endl;
    std::cout << "Avg degree: " << avg_degree << std::endl;

    // Construct the graph
    std::cout << "Constructing graph" << std::endl;
    Graph_t graph(max_degree, points.size());
    parlay::parallel_for(0, points.size(), [&](size_t i) {
        graph[i].clear_neighbors();
        for (size_t j = 0; j < adjlists[i].size(); j++) {
            graph[i].append_neighbor(adjlists[i][j]);
        }
    });
    graph.save(("/ssd1/richard/navgraphs/" + test + ".graph").data());

    // Test QPS/recall
    std::cout << "Testing recall" << std::endl;
    // auto [avg_deg, max_deg] = parlayANN::graph_stats_(graph);
    // parlayANN::Graph_ G_("GreedySetCover", "", graph.size(), avg_deg, max_deg, 0);
    // search_and_parse(G_, graph, points, queries, groundtruth, ("/ssd1/richard/navgraphs/logs/" + test + ".log").data(), 1, true);

    timer.start();
    auto results = parlay::tabulate(queries.size(), [&](size_t i) {
        auto [neighbor, dist_comps] = greedy_search(adjlists, points, 0, i);
        return std::make_pair(neighbor, dist_comps);
    });
    double query_time = timer.next_time();

    // for (size_t i = 0; i < results.size(); i++) {
    //     if (results[i].first != i) {
    //         std::cout << "Query " << i << " returned " << results[i].first << std::endl;
    //         std::cout << "Adjacency list of " << results[i].first << ": ";
    //         for (size_t j = 0; j < adjlists[results[i].first].size(); j++) {
    //             std::cout << adjlists[results[i].first][j] << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    std::cout << "Recall: " << parlay::reduce(parlay::tabulate(queries.size(), [&](size_t i) {
        return results[i].first == i ? 1.0 : 0.0;
    })) / (double)queries.size() << std::endl;
    std::cout << "Avg distance comparisons: " << parlay::reduce(parlay::tabulate(queries.size(), [&](size_t i) {
        return results[i].second;
    })) / (double)queries.size() << std::endl;
    std::cout << "Query time: " << query_time << " seconds" << std::endl;
    std::cout << "Avg QPS: " << queries.size() / query_time << std::endl;

    return 0;
}