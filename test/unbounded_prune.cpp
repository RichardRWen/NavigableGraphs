#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>
#include <mutex>
#include <getopt.h>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/internal/get_time.h>

#include "utils/types.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"

#include "point_set.h"
#include "greedy_search.h"

struct arguments {
    std::string base_path;
    std::string query_path;
    size_t sample_size;
};

void parse_arguments(int argc, char *argv[], arguments &args) {
    struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"base_path", required_argument, NULL, 'b'},
        {"query_path", required_argument, NULL, 'q'},
        {"sample_size", required_argument, NULL, 's'},
        {NULL, 0, NULL, 0}
    };

    args.base_path = "/ssd1/richard/navgraphs/sift_10K.fbin";
    args.query_path = "/ssd1/richard/navgraphs/sift_10K.fbin";
    args.sample_size = -1ULL;

    int opt;
    while ((opt = getopt_long(argc, argv, "hb:q:s:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                std::cout << "Usage: ./prune_neighborhood [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -h, --help                     Show this help message" << std::endl;
                std::cout << "  -b, --base_path <path>         Path to the base dataset" << std::endl;
                std::cout << "  -q, --query_path <path>        Path to the query dataset" << std::endl;
                std::cout << "  -s, --sample_size <size>       Number of points to sample from the dataset" << std::endl;
                exit(EXIT_SUCCESS);
            case 'b':
                args.base_path = std::string(optarg);
                break;
            case 'q':
                args.query_path = std::string(optarg);
                break;
            case 's':
                args.sample_size = std::stoull(optarg);
                break;
            default:
                std::cerr << "Invalid option." << std::endl;
                exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[]) {
    arguments args;
    parse_arguments(argc, argv, args);

    using index_t = uint32_t;
    using value_t = float;
    using PointType = parlayANN::Euclidian_Point<value_t>;
    using PointRangeType = parlayANN::PointRange<PointType>;

    PointSet points(args.base_path.data(), args.sample_size);
    PointSet queries(args.query_path.data());

    size_t progress = 0, max_progress = points.size();
    std::mutex progress_lock;
    parlay::internal::timer timer;
    timer.start();
    auto neighbors = parlay::tabulate(points.size(), [&](size_t v) {
        std::vector<index_t> candidates;
        candidates.reserve(points.size() - 1);
        for (size_t j = 0; j < points.size(); j++) {
            if (j != v) candidates.push_back(j);
        }
        std::vector<value_t> distances;
        distances.reserve(points.size());
        for (size_t j = 0; j < points.size(); j++) {
            distances.push_back(points[v].distance(points[j]));
        }

        std::sort(candidates.begin(), candidates.end(), [&](index_t a, index_t b) {
            return distances[a] < distances[b];
        });

        std::vector<index_t> curr_neighbors;
        for (index_t u : candidates) {
            if (u == v) continue;

            bool add = true;
            for (index_t w : curr_neighbors) {
                value_t vw_dist = distances[w];
                value_t uw_dist = points[u].distance(points[w]);
                if (uw_dist < vw_dist) {
                    add = false;
                    break;
                }
            }

            if (add) curr_neighbors.push_back(u);
        }

        std::lock_guard lock(progress_lock);
        std::cout << "\rProgress: " << ++progress << "/" << max_progress << std::flush;
        return curr_neighbors;
    });
    std::cout << std::endl;
    std::cout << "Computed neighbors in " << timer.next_time() << " seconds" << std::endl;

    auto degrees = parlay::map(neighbors, [](auto &n) { return n.size(); });
    size_t min_degree = parlay::reduce(degrees, parlay::minm<size_t>());
    size_t max_degree = parlay::reduce(degrees, parlay::maxm<size_t>());
    double avg_degree = parlay::reduce(degrees, parlay::addm<size_t>()) / (double)points.size();
    std::cout << "Min degree: " << min_degree << std::endl;
    std::cout << "Max degree: " << max_degree << std::endl;
    std::cout << "Avg degree: " << avg_degree << std::endl;

    timer.start();
    auto results = parlay::tabulate(points.size(), [&](size_t i) {
        auto [neighbor, dist_comps] = greedy_search(neighbors, points, 0, i);
        return std::make_pair(neighbor, dist_comps);
    });
    double query_time = timer.next_time();

    // Compute recall
    
    std::cout << "Avg distance comparisons: " << parlay::reduce(parlay::tabulate(points.size(), [&](size_t i) {
        return results[i].second;
    })) / (double)points.size() << std::endl;
    std::cout << "Query time: " << query_time << " seconds" << std::endl;
    std::cout << "Avg QPS: " << points.size() / query_time << std::endl;
}