#include <iostream>
#include <cstdint>
#include <cstring>

#include <parlay/sequence.h>
#include <parlay/internal/get_time.h>

#include <utils/types.h>
#include <utils/graph.h>
#include <utils/euclidian_point.h>
#include <utils/point_range.h>
#include <utils/parse_results.h>
#include <utils/check_nn_recall.h>

#include "distance_matrix.h"
#include "set_cover.h"

int main(int argc, char* argv[]) {
    std::string test = "sift_1K";
    if (argc > 1) {
        test = argv[1];
    }

    using index_t = uint32_t;
    using value_t = float;
    using Point_t = parlayANN::Euclidian_Point<value_t>;
    using PointRange_t = parlayANN::PointRange<Point_t>;
    using GroundTruth_t = parlayANN::groundTruth<index_t>;
    using Graph_t = parlayANN::Graph<index_t>;

    // Load the points
    std::cout << "Loading test: " << test << std::endl;
    PointRange_t points(("/ssd1/richard/navgraphs/" + test + ".fbin").data());
    PointRange_t queries(("/ssd1/richard/navgraphs/" + test + ".fbin").data());
    GroundTruth_t groundtruth(("/ssd1/richard/navgraphs/" + test + ".gt").data());

    // Compute the adjacency lists
    std::cout << "Computing adjacency lists using greedy set cover" << std::endl;
    parlay::internal::timer timer;
    timer.start();
    SetCoverAdjlists<value_t> set_cover(points);
    auto adjlists = set_cover.adjlists_greedy();
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

    // Test QPS/recall
    std::cout << "Testing recall" << std::endl;
    auto [avg_deg, max_deg] = parlayANN::graph_stats_(graph);
    parlayANN::Graph_ G_("GreedySetCover", "", graph.size(), avg_deg, max_deg, 0);
    search_and_parse(G_, graph, points, queries, groundtruth, ("/ssd1/richard/navgraphs/logs/" + test + ".log").data(), 10, true);

    return 0;
}