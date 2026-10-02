// Linux build (ABI version 3; rebuild the shared library with this wrapper):
// g++ -std=c++11 -fopenmp -O2 -Wall -shared -fPIC -o PAD_on_sphere_Cxx_shared_library.so CC_PAD_on_sphere_python_lib.cc
#include <iostream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <random>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <cstdio>

using namespace std;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)
#define FU __func__
// Wrapper-local policy: PAD checks throw rather than exiting the Python process.
#define ERRORIF(x) do { if (x) throw std::runtime_error(std::string(AT) + ": " #x); } while (false)
// Minimal utilities required by the included PAD implementation. Keep these
// local to this translation unit; utility failures throw, never exit Python.
namespace {
double deg2rad(double degrees)
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    return degrees * pi / 180.0;
}

void spherical_to_cartesian_coordinates(double lat, double lon, double radius,
                                        double &x, double &y, double &z)
{
    x = radius * std::cos(lon) * std::cos(lat);
    y = radius * std::sin(lon) * std::cos(lat);
    z = radius * std::sin(lat);
}

std::mt19937 rng_state_global;

double ran2(long *seed)
{
    if (*seed < 0)
    {
        if (*seed == std::numeric_limits<long>::min())
            throw std::invalid_argument("Random seed cannot be LONG_MIN.");
        rng_state_global.seed(-*seed);
        *seed = -*seed;
    }
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng_state_global);
}

// Used by PAD's optional subsampling helper. Retain the original draw order.
void generate_random_binary_vector_with_fixed_number_of_1_values(
    long p, long count, vector<double> &values, long &seed)
{
    if (p < 0 || count < 0 || p > count)
        throw std::invalid_argument("Invalid subsampling counts.");
    if (count == 0)
    {
        values.clear();
        return;
    }
    const bool sparse = static_cast<double>(p) / static_cast<double>(count) < 0.5;
    const double initial = sparse ? 0.0 : 1.0;
    const double selected = sparse ? 1.0 : 0.0;
    const long draws = sparse ? p : count - p;
    values.assign(static_cast<size_t>(count), initial);
    long changed = 0;
    while (changed < draws)
    {
        const double position = std::floor(ran2(&seed) * static_cast<double>(count));
        if (position < 0 || position >= static_cast<double>(count))
            throw std::runtime_error("Random subsampling index out of range.");
        const size_t index = static_cast<size_t>(position);
        if (values[index] == initial)
        {
            values[index] = selected;
            ++changed;
        }
    }
    long ones = 0;
    for (double value : values)
        if (value == 1.0) ++ones;
    if (ones != p)
        throw std::runtime_error("Incorrect subsampling result count.");
}
}

#include "CU_PAD_on_sphere_code.cc"

namespace {
thread_local char last_error[1024] = {};
// ctypes can release the GIL: serialize access to PAD's shared random generator.
std::mutex calculation_mutex;

void save_error(const char *message) noexcept
{
    std::snprintf(last_error, sizeof(last_error), "%s", message ? message : "Unknown PAD error");
}

vector<vector<double>> make_points(const double *lat, const double *lon,
                                   const double *values, size_t size)
{
    if (!lat || !lon || !values || size == 0)
        throw std::invalid_argument("PAD requires nonempty, non-null input buffers.");
    if (size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        size - 1 > std::numeric_limits<kdtree::IndexType>::max())
        throw std::length_error("Too many PAD grid points.");
    vector<vector<double>> points;
    points.reserve(size);
    for (size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(lat[i]) || lat[i] < -90 || lat[i] > 90 ||
            !std::isfinite(lon[i]) || !std::isfinite(values[i]) || values[i] < 0)
            throw std::invalid_argument("Invalid latitude, longitude, or amount.");
        double x, y, z;
        // Normalize longitude before conversion to avoid overflow for large angles.
        spherical_to_cartesian_coordinates(deg2rad(lat[i]),
            deg2rad(std::remainder(lon[i], 360.0)), Earth_radius, x, y, z);
        points.push_back({x, y, z, values[i]});
    }
    return points;
}

double *pack_results(const PADResults &results, const vector<double> &remaining1,
                     const vector<double> &remaining2)
{
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (remaining1.size() > limit || remaining2.size() > limit - remaining1.size())
        throw std::length_error("PAD output is too large.");
    const size_t tail = remaining1.size() + remaining2.size();
    if (results.size() > (limit - tail) / 4)
        throw std::length_error("PAD output is too large.");
    std::unique_ptr<double[]> buffer(new double[results.size() * 4 + tail]);
    size_t position = 0;
    for (const PADResult &row : results)
    {
        buffer[position++] = row.distance; // Great-circle metres.
        buffer[position++] = row.attributed_amount;
        buffer[position++] = static_cast<double>(row.index1);
        buffer[position++] = static_cast<double>(row.index2);
    }
    for (double value : remaining1) buffer[position++] = value;
    for (double value : remaining2) buffer[position++] = value;
    return buffer.release();
}

// All exceptions from calculations/packing are contained inside the C boundary.
// Callers must supply accessible buffers of the stated lengths.
double *calculate(const double *lat1, const double *lon1, const double *values1, size_t size1,
                  const double *lat2, const double *lon2, const double *values2, size_t size2,
                  size_t *number_of_attributions, double great_circle_cutoff,
                  int64_t random_seed, bool same_grid) noexcept
{
    last_error[0] = '\0';
    if (number_of_attributions) *number_of_attributions = 0;
    try
    {
        if (!number_of_attributions)
            throw std::invalid_argument("Missing attribution-count output pointer.");
        std::lock_guard<std::mutex> lock(calculation_mutex);
        validate_PAD_cutoff(great_circle_cutoff);
        validate_PAD_random_seed(random_seed);
        auto points1 = make_points(lat1, lon1, values1, size1);
        vector<vector<double>> points2;
        if (same_grid)
        {
            if (!values2 || size2 != size1)
                throw std::invalid_argument("Same-grid fields require matching non-null input buffers.");
            points2.reserve(size2);
            for (size_t i = 0; i < size2; ++i)
            {
                if (!std::isfinite(values2[i]) || values2[i] < 0)
                    throw std::invalid_argument("Invalid amount in the second field.");
                // Reuse the exact XYZ values already calculated for this grid.
                // Only the precipitation amount differs between the two rows.
                const auto &point = points1[i];
                points2.push_back({point[0], point[1], point[2], values2[i]});
            }
        }
        else
            points2 = make_points(lat2, lon2, values2, size2);
        vector<double> remaining1, remaining2;
        PADResults results = same_grid
            ? calculate_PAD_results_assume_same_grid_and_remove_overlap(
                points1, points2, great_circle_cutoff, remaining1, remaining2, random_seed)
            : calculate_PAD_results_assume_different_grid(
                points1, points2, great_circle_cutoff, remaining1, remaining2, random_seed);
        double *buffer = pack_results(results, remaining1, remaining2);
        *number_of_attributions = results.size();
        return buffer;
    }
    catch (const std::exception &e) { save_error(e.what()); }
    catch (...) { save_error("Unknown C++ exception during PAD calculation."); }
    return nullptr;
}
}

extern "C" int PAD_wrapper_abi_version() noexcept { return 3; }
// Error text is owned by the library, valid until the next call on this thread.
extern "C" const char *PAD_last_error() noexcept { return last_error; }
extern "C" void free_mem_double_array(double *buffer) noexcept { delete[] buffer; }

// ABI v3: cutoff is great-circle distance in metres; -1 selects a random seed.
extern "C" double *calculate_PAD_results_assume_same_grid_ctypes(
    const double *lat, const double *lon, const double *values1, const double *values2,
    size_t size, size_t *count, double great_circle_cutoff, int64_t random_seed) noexcept
{
    return calculate(lat, lon, values1, size, lat, lon, values2, size,
                     count, great_circle_cutoff, random_seed, true);
}

extern "C" double *calculate_PAD_results_assume_different_grid_ctypes(
    const double *lat1, const double *lon1, const double *values1, size_t size1,
    const double *lat2, const double *lon2, const double *values2, size_t size2,
    size_t *count, double great_circle_cutoff, int64_t random_seed) noexcept
{
    return calculate(lat1, lon1, values1, size1, lat2, lon2, values2, size2,
                     count, great_circle_cutoff, random_seed, false);
}
