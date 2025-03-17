#include <iostream>
#include <cstdint>
#include <cstring>
#include <getopt.h>

#include <parlay/sequence.h>
#include <parlay/primitives.h>
#include <parlay/internal/get_time.h>

#include <utils/types.h>
#include <utils/graph.h>

#include "point_set.h"
#include "greedy_search.h"

struct arguments {
    std::string graph_file;
    std::string base_file;
    std::string query_file;
    std::string ground_truth_file;
    size_t k;
};

void print_args(arguments &args) {
    std::cout << "Graph file: " << args.graph_file << std::endl;
    std::cout << "Base file: " << args.base_file << std::endl;
    std::cout << "Query file: " << args.query_file << std::endl;
    if (!args.ground_truth_file.empty()) {
        std::cout << "Ground truth file: " << args.ground_truth_file << std::endl;
    }
    std::cout << "k: " << args.k << std::endl;
}

void print_usage(char *progname) {
    std::cerr << "Usage: " << progname << " [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  -g, --graph <file>           Graph file\n";
    std::cerr << "  -b, --base <file>            Base file\n";
    std::cerr << "  -q, --query <file>           Query file\n";
    std::cerr << "  -t, --ground_truth <file>    Ground truth file\n";
    std::cerr << "  -k, --k <int>                Number of neighbors to search for\n";
    std::cerr << "  -h, --help                   Print this help message\n";
}

void parse_args(int argc, char *argv[], arguments &args) {
    static struct option long_options[] = {
        {"graph", required_argument, 0, 'g'},
        {"base", required_argument, 0, 'b'},
        {"query", required_argument, 0, 'q'},
        {"ground_truth", required_argument, 0, 't'},
        {"k", required_argument, 0, 'k'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    args.graph_file = "/ssd1/richard/navgraphs/sift_10K.graph";
    args.base_file = "/ssd1/richard/navgraphs/sift_10K.fbin";
    args.query_file = "/ssd1/richard/navgraphs/sift_10K.fbin";
    args.ground_truth_file = "";
    args.k = 1;

    int c;
    while ((c = getopt_long(argc, argv, "g:b:q:t:k:h", long_options, NULL)) != -1) {
        switch (c) {
            case 'g':
                args.graph_file = optarg;
                break;
            case 'b':
                args.base_file = optarg;
                break;
            case 'q':
                args.query_file = optarg;
                break;
            case 't':
                args.ground_truth_file = optarg;
                break;
            case 'k':
                args.k = std::atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
}

int main(int argc, char *argv[]) {
    arguments args;
    parse_args(argc, argv, args);
    print_args(args);

    // Load graph
    parlayANN::Graph<uint32_t> graph(args.graph_file.data());
    std::cout << "Loaded graph with " << graph.size() << " vertices" << std::endl;

    // Load points
    PointSet points(args.base_file.data(), graph.size());
    PointSet queries = (args.query_file == args.base_file)
        ? PointSet(args.base_file.data(), graph.size())
        : PointSet(args.query_file.data());
    std::cout << "Loaded " << points.size() << " points" << std::endl;
    std::cout << "Loaded " << queries.size() << " queries" << std::endl;

    // Compute the adjacency lists
    parlay::sequence<std::vector<uint32_t>> adjlists = parlay::tabulate(graph.size(), [&](size_t i) {
        std::vector<uint32_t> adjlist;
        for (size_t j = 0; j < graph[i].size(); j++) {
            adjlist.push_back(graph[i][j]);
        }
        return adjlist;
    });

    // Compute the ground truth
    parlay::sequence<parlay::sequence<uint32_t>> ground_truth;
    if (!args.ground_truth_file.empty()) {
        std::cout << "Loading ground truth from " << args.ground_truth_file << std::endl;
        parlayANN::groundTruth<uint32_t> gt(args.ground_truth_file.data());
        if (gt.size() != queries.size()) {
            std::cerr << "Error: ground truth size does not match query size" << std::endl;
            return 1;
        }
        ground_truth = parlay::tabulate(queries.size(), [&](size_t i) {
            parlay::sequence<uint32_t> neighbors;
            neighbors.reserve(args.k);
            for (size_t j = 0; j < args.k; j++) {
                neighbors.push_back(gt.coordinates(i, j));
            }
            return neighbors;
        });
    }
    else {
        std::cout << "Computing ground truth" << std::endl;
        ground_truth = parlay::tabulate(queries.size(), [&](size_t i) {
            auto distances = parlay::tabulate(points.size(), [&](size_t j) {
                return points[j].distance(queries[i]);
            });
            auto indices = parlay::tabulate(points.size(), [&](size_t j) {
                return j;
            });
            std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
                return distances[a] < distances[b];
            });

            parlay::sequence<uint32_t> neighbors(args.k);
            std::copy(indices.begin(), indices.begin() + args.k, neighbors.begin());
            return neighbors;
        });
    }

    // Perform queries
    parlay::internal::timer timer;
    timer.start();
    auto results = parlay::tabulate(points.size(), [&](size_t i) {
        auto [neighbor, dist_comps] = greedy_search(adjlists, points, 0, i);
        return std::make_pair(neighbor, dist_comps);
    });
    double query_time = timer.next_time();

    // Compute recall
    size_t correct = 0;
    for (size_t i = 0; i < queries.size(); i++) {
        for (size_t j = 0; j < args.k; j++) {
            if (results[i].first == ground_truth[i][j]) {
                correct++;
                break;
            }
        }
    }
    double recall = correct / (double)queries.size() / args.k;

    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Avg distance comparisons: " << parlay::reduce(parlay::tabulate(points.size(), [&](size_t i) {
        return results[i].second;
    })) / (double)points.size() << std::endl;
    std::cout << "Query time: " << query_time << " seconds" << std::endl;
    std::cout << "Avg QPS: " << points.size() / query_time << std::endl;

    return 0;
}