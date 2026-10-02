#include "CU_kdtree_with_index.cc"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

const double Earth_radius= 6371*1000;
constexpr size_t PAD_dimensions = 3;
using PADPoint = kdtree::Point_str<PAD_dimensions>;
using PADKdTree = kdtree::KdTree<PAD_dimensions>;

// Distance, attributed amount, first-field index, second-field index.
struct PADResult
{
    // Metres: Euclidean during attribution; great-circle in completed results.
    double distance;
    double attributed_amount;
    std::size_t index1;
    std::size_t index2;
};
using PADResults = std::vector<PADResult>;

// Optional calculation seed: -1 (the default) chooses a fresh seed using
// std::random_device; 0..UINT32_MAX supplies an explicit seed. Both reseed the
// generator before either index shuffle. ran2 uses the
// same generator, so this also controls random choices during attribution.
// Reproducibility assumes identical inputs and the same library implementation.
// This uses/advances the shared RNG and does not make PAD calls thread-safe.
void validate_PAD_random_seed(const std::int64_t random_seed)
	{
	if (random_seed < -1 || random_seed > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()))
		throw std::invalid_argument("PAD random seed must be -1 (random) or an unsigned 32-bit value.");
	}

template<typename T, size_t Dimensions>
double squared_euclidian_distance_multidimensional(const array<T, Dimensions> &p1, const array<T, Dimensions> &p2)
	{
	double sum=0;
	for (size_t il=0; il < Dimensions; il++)
		sum+=(p1[il]-p2[il])*(p1[il]-p2[il]);
	return(sum);
	}

double great_circle_distance_to_euclidian_distance(const double great_circle_distance)
	{
	return(2*Earth_radius*sin(great_circle_distance/(2*Earth_radius)));
	}

double euclidian_distance_to_great_circle_distance(const double euclidian_distance)
	{
	if (!std::isfinite(euclidian_distance) || euclidian_distance < 0)
		throw std::domain_error("Euclidean distance must be finite and non-negative.");

	double asin_argument = euclidian_distance/(2*Earth_radius);
	// Coordinates and their squared differences are rounded in PointType.
	// Allow a small overshoot at antipodal points, but do not hide invalid distances.
	const double rounding_tolerance = 8.0 * std::numeric_limits<kdtree::PointType>::epsilon();
	if (asin_argument > 1.0)
		{
		if (asin_argument > 1.0 + rounding_tolerance)
			throw std::domain_error("Euclidean distance exceeds the Earth's diameter beyond rounding tolerance.");
		asin_argument = 1.0;
		}
	return(2*Earth_radius*asin(asin_argument));
	}

// Apply exactly once after attribution, before returning results to consumers.
void convert_PAD_results_euclidian_distances_to_great_circle_distances(PADResults &results)
	{
	for (auto &result : results)
		// Overlap records already have the correct great-circle distance of zero.
		if (result.distance != 0)
			result.distance = euclidian_distance_to_great_circle_distance(result.distance);
	}

// Validate once during input preparation, not on every nearest-neighbor query.
void validate_PAD_point(const vector<double> &point)
	{
	if (point.size() != PAD_dimensions + 1)
		throw std::invalid_argument("PAD points must contain x, y, z and an amount.");
	for (size_t axis = 0; axis < PAD_dimensions; ++axis)
		if (!std::isfinite(point[axis]) ||
			std::fabs(point[axis]) > std::numeric_limits<kdtree::PointType>::max())
			throw std::domain_error("PAD coordinates must be finite and representable in PointType.");
	if (!std::isfinite(point.back()) || point.back() < 0)
		throw std::domain_error("PAD amounts must be finite and non-negative.");
	}

void validate_PAD_cutoff(const double euclidian_attribution_distance_cutoff)
	{
	if (!std::isfinite(euclidian_attribution_distance_cutoff) || euclidian_attribution_distance_cutoff < 0)
		throw std::domain_error("PAD cutoff must be finite and non-negative.");
	}

// Public cutoffs are shortest great-circle distances in metres. At or beyond
// half the circumference every pair is allowed. Use an unrestricted internal
// cutoff there so float-coordinate rounding cannot exclude antipodal pairs.
double PAD_great_circle_cutoff_to_euclidian(const double great_circle_attribution_distance_cutoff)
	{
	validate_PAD_cutoff(great_circle_attribution_distance_cutoff);
	if (great_circle_attribution_distance_cutoff >= std::acos(-1.0) * Earth_radius)
		return std::numeric_limits<double>::max();
	return great_circle_distance_to_euclidian_distance(great_circle_attribution_distance_cutoff);
	}

// Retain every point in input order, so result indices need no remapping.
vector <vector <double> > convert_latlon_points_to_3D_points(const vector <vector <double> > &points_lat_lon)
	{
	vector <vector <double> > points;
	for ( unsigned long il=0; il < points_lat_lon.size(); il++)
		{
		const auto &row = points_lat_lon[il];
		if (row.size() != 3)
			throw std::invalid_argument("Latitude/longitude points must contain latitude, longitude and an amount.");
		if (!std::isfinite(row[0]) || !std::isfinite(row[1]) ||
			!std::isfinite(row[2]) || row[2] < 0)
			throw std::domain_error("Latitude/longitude and amounts must be finite; amounts must be non-negative.");
		double x,y,z;
		spherical_to_cartesian_coordinates(deg2rad(points_lat_lon[il][0]), deg2rad(points_lat_lon[il][1]), Earth_radius, x, y, z);
		points.push_back({x,y,z,points_lat_lon[il][2]});
		}
	return(points);
	}


PADPoint kdtree_Point_str_from_point(const vector <double> &point, const size_t ind)
	{
	ERRORIF(point.size() != PAD_dimensions + 1);
	PADPoint p;
	p.coords = {{static_cast<kdtree::PointType>(point[0]),
		static_cast<kdtree::PointType>(point[1]),
		static_cast<kdtree::PointType>(point[2])}};
	p.set_index(ind);
	return(p);
	}

// The output record is valid only when this returns true; cutoff-only removals return false.
// The cutoff is squared Euclidean chord distance (m^2); the output distance is Euclidean (m).
bool perform_one_PAD_iteration_with_attribution_distance_cutoff(const vector <vector <double>> &points1, const vector <vector <double>> &points2, vector <double> &values1, vector <double> &values2, vector <size_t> &index_list1, vector <size_t> &index_list2, PADKdTree &kdtree1, PADKdTree &kdtree2, long &idum, const bool f1_is_fa, const double squared_euclidian_attribution_distance_cutoff, PADResult &result)
	{
	// choose the last point from list1
	const auto ind1=index_list1.back();
	const PADPoint p1 = kdtree_Point_str_from_point(points1[ind1], ind1);

	// find the closes non-zero point in the other field
	//auto begin = std::chrono::high_resolution_clock::now();
	const auto node = kdtree2.findNearestNode_in_radius(p1, squared_euclidian_attribution_distance_cutoff);
	//temp.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9);
	if (node == nullptr)
		{
		// No acceptable nearest neighbor: keep the amount unattributed.
		kdtree1.deleteNode(p1);
		index_list1.pop_back();
		while (!index_list1.empty() && values1[index_list1.back()] == 0)
			index_list1.pop_back();
		while (!index_list2.empty() && values2[index_list2.back()] == 0)
			index_list2.pop_back();
		return false;
		}
	const PADPoint p2 = node->val;
	const auto ind2=p2.index;

	const double min_squared_euclidian_distance=squared_euclidian_distance_multidimensional(p1.coords, p2.coords);

	//cout << ind1 << " " << values1[ind1] << endl;
	//cout << ind2 << " " << values2[ind2] << endl;

	bool has_result = false;

	// Squared Euclidean distance larger than the squared Euclidean cutoff.
	if (min_squared_euclidian_distance >  squared_euclidian_attribution_distance_cutoff)
		{
		// remove the point from the kdtree and index
		kdtree1.deleteNode(p1);
		index_list1.pop_back();

		//cout << index_list1.size() << " " << index_list2.size() << " " << sqrt(min_squared_euclidian_distance) << endl;

		}

	else
		{
			// value reduction
		const double value_reduction= min(values1[ind1],values2[ind2]);
		values1[ind1]-=value_reduction;
		values2[ind2]-=value_reduction;

		//cout << list_nzp_f1.size() << " " << list_nzp_f2.size() << endl;

		// if necessary remove the point from kdtree1
		if (values1[ind1] == 0)
			kdtree1.deleteNode(p1);
		// else swap with random point so this is not the last point anymore - so it is not automatically chosen the next time
		else
			swap(index_list1.back(),index_list1[ floor(ran2(&idum) * (double)index_list1.size()) ]);

		// if necessary remove the point from kdtree2
		if (values2[ind2] == 0)
			kdtree2.deleteNode(p2);

		//cout << list_nzp_f1.size() << " " << list_nzp_f2.size() << endl;



		result.distance = sqrt(min_squared_euclidian_distance);
		result.attributed_amount = value_reduction;
		if (f1_is_fa)
			{
			result.index1 = ind1;
			result.index2 = ind2;
			}
		else
			{
			result.index1 = ind2;
			result.index2 = ind1;
			}
		has_result = true;

		//cout << index_list1.size() << " " << index_list2.size() << endl;
		}

	// remove all zero points at the end of list
	while (index_list1.size() > 0 && values1[index_list1.back()] == 0)
		index_list1.pop_back();

	// remove all zero points at the end of vector
	while (index_list2.size() > 0 && values2[index_list2.back()] == 0)
		index_list2.pop_back();

		//cout << index_list1.size() << " " << index_list2.size() << endl;

	return(has_result);

	}



// Validate complete rows before reading amounts, including zero-valued rows.
vector<double> check_points(const vector<vector<double>> &points)
	{
	size_t count=0;
	vector<double> values;
	for (size_t il=0; il < points.size(); il++)
		{
		validate_PAD_point(points[il]);
		const double val=points[il].back();
		values.push_back(val);
		if (val > 0) count++;
		}
	if (count == 0)
		throw std::domain_error("PAD requires at least one positive amount in each field.");
	return(values);
	}

// Check points, subsample if requested, then normalize without an unscaled sum.
vector<double> check_points_subsample_and_normalize(const vector<vector<double>> &points, const long max_number_of_nonzero_points)
	{
	vector<double> values = check_points(points);
	size_t count=0;
	for (double value : values)
		if (value > 0) count++;

	if (max_number_of_nonzero_points > 0 && count > static_cast<size_t>(max_number_of_nonzero_points))
		{
		// The sampling utility uses signed long counts.
		if (count > static_cast<size_t>(std::numeric_limits<long>::max()))
			throw std::length_error("Too many points for the subsampling utility.");
		vector<double> v;
		long idum=1;
		generate_random_binary_vector_with_fixed_number_of_1_values(max_number_of_nonzero_points, count, v, idum);

		size_t counter=0;
		for (size_t il=0; il < values.size(); il++)
			if (values[il] > 0)
				{
				if (v[counter] == 0) values[il]=0;
				counter++;
				}
		cout << "Notice: Subsampling " << max_number_of_nonzero_points << " nonzero points out of " << count << "." << endl;
		}

	double scale=0;
	for (double value : values)
		if (value > scale) scale=value;
	if (scale == 0)
		throw std::domain_error("Cannot normalize amounts with no positive retained value.");

	// Every term is in [0,1]; at least one is exactly 1.
	double scaled_sum=0;
	for (double value : values)
		scaled_sum += value/scale;
	for (double &value : values)
		value = (value/scale)/scaled_sum;
	return(values);
	}


// from non-zero points construct kdtree and shuffled index list
void construct_kdtree_with_shuffled_index_list(const vector <vector <double>> &points, const vector <double> &values, vector <size_t> &index_list, PADKdTree &kdtree)
	{
	if (points.size() != values.size())
		throw std::invalid_argument("PAD point and amount arrays must have matching sizes.");
	index_list.clear();
	ERRORIF(kdtree.is_empty() == false);

	vector<PADPoint> pointsX;
	for (size_t ip=0; ip < points.size(); ip++)
		if (values[ip] > 0)
			{
			pointsX.push_back(kdtree_Point_str_from_point(points[ip], ip));
			index_list.push_back(ip);
			}

	// shuffle list
	std::shuffle(index_list.begin(), index_list.end(), rng_state_global);

	// construct kdtree
	if (!kdtree.buildKdTree(pointsX))
		{
		index_list.clear();
		throw std::runtime_error("PAD k-d tree construction failed: empty or unsupported point count.");
		}
	}


void reserve_PAD_results_for_attribution(PADResults &results, const size_t active_points1, const size_t active_points2)
	{
	if (active_points1 == 0 || active_points2 == 0) return;
	// Each iteration removes at least one active point, so this bounds the
	// additional rows, including when some removals produce no result.
	const size_t available = results.max_size() - results.size();
	if (active_points1 > available || active_points2 > available - active_points1)
		throw std::length_error("PAD result capacity exceeds vector::max_size().");
	results.reserve(results.size() + active_points1 + active_points2);
	}


// Shared attribution engine. Values are already validated/prepared, with at
// least one positive value in each field. Append Euclidean-distance records to
// out (which may already contain overlap records); wrappers convert to GCD once.
// All large containers are passed by reference; no field-state aggregation.
void calculate_PAD_results_general(const vector<vector<double>> &points1, const vector<vector<double>> &points2, const double euclidian_attribution_distance_cutoff, vector<double> &values1, vector<double> &values2, PADResults &out, const std::int64_t random_seed = -1)
	{
	if (&values1 == &values2)
		throw std::invalid_argument("PAD requires distinct output amount vectors for the two fields.");
	validate_PAD_cutoff(euclidian_attribution_distance_cutoff);
	validate_PAD_random_seed(random_seed);
	std::uint32_t selected_seed;
	if (random_seed == -1)
		{
		std::random_device seed_source;
		selected_seed = std::uniform_int_distribution<std::uint32_t>(
			0, std::numeric_limits<std::uint32_t>::max())(seed_source);
		}
	else
		selected_seed = static_cast<std::uint32_t>(random_seed);
	rng_state_global.seed(selected_seed);
	// Record both automatic and explicit seeds for reproducibility.
	std::cout << "PAD random seed: " << selected_seed << std::endl;
	// A huge finite cutoff means unrestricted distance; avoid overflowing its square.
	const double squared_euclidian_attribution_distance_cutoff =
		euclidian_attribution_distance_cutoff > std::sqrt(std::numeric_limits<double>::max())
		? std::numeric_limits<double>::infinity()
		: euclidian_attribution_distance_cutoff * euclidian_attribution_distance_cutoff;

	auto begin = std::chrono::high_resolution_clock::now();
	vector <size_t> index_list1;
	PADKdTree kdtree1;
	construct_kdtree_with_shuffled_index_list(points1, values1, index_list1, kdtree1);
	vector <size_t> index_list2;
	PADKdTree kdtree2;
	construct_kdtree_with_shuffled_index_list(points2, values2, index_list2, kdtree2);
	cout << "----- kdtree construction: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	long idum=1;
	bool fa_turn=true;

	begin = std::chrono::high_resolution_clock::now();
	reserve_PAD_results_for_attribution(out, index_list1.size(), index_list2.size());
	while (index_list1.size() > 0 && index_list2.size() > 0)
		{
		PADResult result;
		bool has_result = false;
		if (fa_turn)
			{
			has_result = perform_one_PAD_iteration_with_attribution_distance_cutoff(points1, points2, values1, values2, index_list1, index_list2, kdtree1, kdtree2, idum, fa_turn, squared_euclidian_attribution_distance_cutoff, result);
			fa_turn=false;
			}
		else
			{
			has_result = perform_one_PAD_iteration_with_attribution_distance_cutoff(points2, points1, values2, values1, index_list2, index_list1, kdtree2, kdtree1, idum, fa_turn, squared_euclidian_attribution_distance_cutoff, result);
			fa_turn=true;
			}

		if (has_result)
			out.push_back(result);
		}

	kdtree1.free_memory();
	kdtree2.free_memory();

	cout << "----- attribution: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	}


PADResults calculate_PAD_results_assume_same_grid_and_remove_overlap(const vector <vector <double> > &points1, const vector <vector <double> > &points2, const double great_circle_attribution_distance_cutoff, vector <double> &values1, vector <double> &values2, const std::int64_t random_seed = -1)
	{
	// Check before assigning either output or removing overlap.
	if (&values1 == &values2)
		throw std::invalid_argument("PAD requires distinct output amount vectors for the two fields.");
	const double euclidian_attribution_distance_cutoff =
		PAD_great_circle_cutoff_to_euclidian(great_circle_attribution_distance_cutoff);
	validate_PAD_random_seed(random_seed);

	ERRORIF(points1.size() != points2.size());

	auto begin = std::chrono::high_resolution_clock::now();
	values1 = check_points(points1);
	values2 = check_points(points2);
	cout << "----- preprocessing: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	PADResults out;
	bool any_non_zero_points1= false;
	bool any_non_zero_points2= false;
	for (unsigned long il=0; il < points1.size(); il++)
		{
		double value_reduction= min(values1[il],values2[il]);
		if (value_reduction > 0)
			{
			values1[il]-=value_reduction;
			values2[il]-=value_reduction;
			out.push_back(PADResult{0,value_reduction,il,il});
			}

		if (any_non_zero_points1 == false)
			if (values1[il] > 0) any_non_zero_points1=true;
		if (any_non_zero_points2 == false)
			if (values2[il] > 0) any_non_zero_points2=true;
		}

	// Overlap preprocessing may exhaust either field; do not build/shuffle trees then.
	if (any_non_zero_points1 && any_non_zero_points2)
		calculate_PAD_results_general(points1, points2, euclidian_attribution_distance_cutoff, values1, values2, out, random_seed);

	convert_PAD_results_euclidian_distances_to_great_circle_distances(out);
	return(out);
	}


PADResults calculate_PAD_results_assume_different_grid(const vector <vector <double> > &points1, const vector <vector <double> > &points2, const double great_circle_attribution_distance_cutoff, vector <double> &values1, vector <double> &values2, const std::int64_t random_seed = -1)
	{
	// Check before assigning either output.
	if (&values1 == &values2)
		throw std::invalid_argument("PAD requires distinct output amount vectors for the two fields.");
	const double euclidian_attribution_distance_cutoff =
		PAD_great_circle_cutoff_to_euclidian(great_circle_attribution_distance_cutoff);
	validate_PAD_random_seed(random_seed);
	auto begin = std::chrono::high_resolution_clock::now();
	values1 = check_points(points1);
	values2 = check_points(points2);
	cout << "----- preprocessing: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin).count() * 1e-9 << " s" << endl;

	PADResults out;
	calculate_PAD_results_general(points1, points2, euclidian_attribution_distance_cutoff, values1, values2, out, random_seed);
	convert_PAD_results_euclidian_distances_to_great_circle_distances(out);
	return(out);
	}


double calculate_PAD_from_PAD_results(const PADResults &results)
	{
	double sum_weights=0;
	double sum=0;
	for (unsigned long il=0; il < results.size(); il++)
		{
		sum+=results[il].distance*results[il].attributed_amount;
		sum_weights+=results[il].attributed_amount;
		}

	if (!(sum_weights > 0))
		throw std::domain_error("PAD is undefined without positive attributed weight.");
	return(sum/sum_weights);
	}
