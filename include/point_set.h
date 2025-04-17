#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>

#include <parlay/sequence.h>

template <typename value_t = float>
class PointSet {
public:
    class Point {
    public:
        using distanceType = value_t;
        parlay::sequence<value_t> coords;
        size_t _id;

        Point() {}

        Point(size_t d) : coords(parlay::sequence<value_t>::uninitialized(d)) {}

        Point(size_t id, value_t *coords, size_t d) : _id(id), coords(parlay::sequence<value_t>::uninitialized(d)) {
            std::memcpy(this->coords.begin(), coords, d * sizeof(value_t));
        }

        Point(parlay::sequence<value_t> &&coords) : coords(coords) {}

        value_t operator[](size_t i) const {
            return coords[i];
        }

        size_t id() const {
            return _id;
        }

        size_t size() const {
            return coords.size();
        }

        value_t distance(const Point &other) const {
            value_t dist = 0;
            for (size_t i = 0; i < coords.size(); i++) {
                dist += (coords[i] - other.coords[i]) * (coords[i] - other.coords[i]);
            }
            return dist;
        }

        bool same_as(const Point &other) const {
            return id() == other.id();
        }
    
        void prefetch() const {}
    
        static bool is_metric() { return true; }
    };

    struct parameters {
        int dims;
        int num_bytes() const {return dims * sizeof(value_t);}

        parameters() : dims(0) {}
        parameters(int dims) : dims(dims) {}
    };

    PointSet() : _size(0) {}

    PointSet(const PointSet &other) : _size(other._size), points(other.points) {}

    PointSet(std::string filename, size_t head_size = -1ULL) {
        std::ifstream reader(filename);
        if (!reader.is_open()) {
            std::cout << "Data file " << filename << " not found" << std::endl;
            std::abort();
        }

        uint32_t n, d;
        reader.read((char *)(&n), sizeof(uint32_t));
        reader.read((char *)(&d), sizeof(uint32_t));
        _size = std::min<size_t>(n, head_size);
        params = parameters(d);

        std::vector<value_t> data(n * d);
        reader.read((char *)data.data(), n * d * sizeof(value_t));

        points = parlay::tabulate(_size, [&](size_t i) {
            return Point(i, data.data() + i * d, d);
        });
    }

    Point& operator[](size_t i) {
        return points[i];
    }
    const Point& operator[](size_t i) const {
        return points[i];
    }

    size_t size() const {
        return _size;
    }

    size_t dimension() const {
        return points[0].size();
    }

    parameters params;
private:
    size_t _size;
    size_t dims;
    parlay::sequence<Point> points;
};