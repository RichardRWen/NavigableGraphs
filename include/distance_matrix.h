#pragma once

#include <parlay/parallel.h>

#include <utils/point_range.h>

template <typename val_t>
class SortedDistanceMatrix {
public:
    using Point_t = parlayANN::Euclidian_Point<val_t>;
    using PointRange_t = parlayANN::PointRange<Point_t>;
    using pid_t = std::pair<uint32_t, val_t>;

    PointRange_t &points;
    pid_t *distances;

    SortedDistanceMatrix(PointRange_t &points) : points(points) {
        distances = (pid_t *) malloc(points.size() * points.size() * sizeof(pid_t));

        parlay::parallel_for(0, points.size(), [&](size_t i) {
            distances[i * points.size() + i] = {i, 0};
            for (size_t j = i + 1; j < points.size(); j++) {
                val_t dist = points[i].distance(points[j]);
                distances[i * points.size() + j] = {j, dist};
                distances[j * points.size() + i] = {i, dist};
            }
        }, 1);
        parlay::parallel_for(0, points.size(), [&](size_t i) {
            std::sort(distances + i * points.size(), distances + (i + 1) * points.size(),
                      [](const pid_t &a, const pid_t &b) { return a.second < b.second; });
        }, 1);
    }

    SortedDistanceMatrix(const SortedDistanceMatrix &) = delete;
    SortedDistanceMatrix &operator=(const SortedDistanceMatrix &) = delete;

    SortedDistanceMatrix(SortedDistanceMatrix &&other) noexcept
        : points(other.points), distances(other.distances) {
        other.distances = nullptr;
    }
    SortedDistanceMatrix &operator=(SortedDistanceMatrix &&other) noexcept {
        if (this != &other) {
            free(distances);
            distances = other.distances;
            other.distances = nullptr;
        }
        return *this;
    }

    ~SortedDistanceMatrix() {
        free(distances);
    }
};