
// MIT License
//
// Copyright (c) 2019 SuYuxi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Comment by Gregor Skok:
// The original code by SuYuxi has beed modified a lot to make it faster and to include new features

#include <vector>
#include <array>
#include <queue>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <type_traits>
#include <climits>

// Byte transport only: these helpers know nothing about k-d tree records.
namespace kdtree_binary_io
{
inline void write_file(const std::string &name, const std::vector<uint8_t> &bytes)
{
    std::ofstream file(name, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Cannot open k-d tree file for writing: " + name);
    for (size_t offset = 0; offset < bytes.size();)
    {
        const size_t count = std::min<size_t>(bytes.size() - offset, 1024 * 1024);
        file.write(reinterpret_cast<const char *>(bytes.data() + offset), static_cast<std::streamsize>(count));
        if (!file) throw std::runtime_error("Cannot write k-d tree file: " + name);
        offset += count;
    }
    file.flush();
    if (!file) throw std::runtime_error("Cannot flush k-d tree file: " + name);
    file.close();
    if (file.fail()) throw std::runtime_error("Cannot close k-d tree file: " + name);
}

inline std::vector<uint8_t> read_file(const std::string &name)
{
    std::ifstream file(name, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open k-d tree file for reading: " + name);
    const std::streamoff length = file.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine k-d tree file size: " + name);
    std::vector<uint8_t> bytes;
    if (static_cast<uintmax_t>(length) > bytes.max_size())
        throw std::length_error("K-d tree file is too large: " + name);
    bytes.resize(static_cast<size_t>(length));
    file.seekg(0, std::ios::beg);
    if (!file) throw std::runtime_error("Cannot seek in k-d tree file: " + name);
    for (size_t offset = 0; offset < bytes.size();)
    {
        const size_t count = std::min<size_t>(bytes.size() - offset, 1024 * 1024);
        file.read(reinterpret_cast<char *>(bytes.data() + offset), static_cast<std::streamsize>(count));
        if (!file) throw std::runtime_error("Cannot read complete k-d tree file: " + name);
        offset += count;
    }
    if (file.peek() != std::char_traits<char>::eof() || file.bad())
        throw std::runtime_error("K-d tree file changed or could not be read: " + name);
    file.clear(); // Clear the expected EOF before checking close().
    file.close();
    if (file.fail()) throw std::runtime_error("Cannot close k-d tree file: " + name);
    return bytes;
}
}

namespace kdtree
	{
	using namespace std;

	template<typename Container>
	static string kdtree_output_vector_as_string(const Container &vec, string separator)
		{
		ostringstream s1;
		for (long il=0; il < (long)vec.size(); il++)
			{
			s1 << vec[il];
			if (il < (long)vec.size() - 1)
				s1 << separator;
			}
		return(s1.str());
		}

	[[noreturn]] inline void throw_kdtree_error(const char * const file, const int line, const char * const function, const string &message)
		{
		throw runtime_error(string(file) + ":" + to_string(line) + " " + function + ": " + message);
		}

#define KDTREE_ERRORIF(condition) \
	do \
		{ \
		if (condition) \
			throw_kdtree_error(__FILE__, __LINE__, __func__, "condition failed: " #condition); \
		} \
	while (false)

	typedef float PointType;
	typedef uint32_t IndexType;
	//typedef vector<PointType> Point; //Presents Point type by vector<int> like (x, y, z)

	template<size_t Dimensions>
	struct Point_str  // Point_str Structure declaration
		{
		static_assert(Dimensions > 0, "A k-d tree must have at least one dimension.");
		array<PointType, Dimensions> coords{};
		IndexType index = 0;

		// Validate the original point identifier, not just the tree's node count.
		void set_index(const size_t index_)
			{
			KDTREE_ERRORIF(index_ > numeric_limits<IndexType>::max());
			index = static_cast<IndexType>(index_);
			}

		void set(const array<PointType, Dimensions> &coords_, const size_t &index_)
			{
			set_index(index_);
			coords = coords_;
			}

		void free_memory()
			{
			// Coordinates are stored inline and do not own dynamic memory.
			}

		string output() const
			{
			ostringstream s1;
			s1 << "(";
			for (unsigned long il=0; il < coords.size(); il++)
				{
				s1 << coords[il];
				if (il< coords.size() - 1)
					s1 << ",";
				}
			s1 << "):" << index;
			return(s1.str());
			}
 		};


	template<size_t Dimensions>
	bool operator==(const Point_str<Dimensions>& lhs, const Point_str<Dimensions>& rhs)
		{
		if (lhs.coords == rhs.coords && lhs.index == rhs.index)
			return(true);
		return(false);
		}

	template<size_t Dimensions>
	bool operator!=(const Point_str<Dimensions>& lhs, const Point_str<Dimensions>& rhs)
		{
		if (lhs.coords != rhs.coords || lhs.index != rhs.index)
			return(true);
		return(false);
		}

	template<size_t Dimensions>
	class KdTree;

	template<size_t Dimensions>
	struct KdTreeNode
		{
		KdTreeNode() = default;

		KdTreeNode(const Point_str<Dimensions> &_val)
			{
			val = _val;
			leftNode = nullptr;
			rightNode = nullptr;
			}

		Point_str<Dimensions> val;
		array<PointType, Dimensions> coords_max{};
		array<PointType, Dimensions> coords_min{};

		private:
		friend class KdTree<Dimensions>;
		KdTreeNode<Dimensions> * leftNode = nullptr;
		KdTreeNode<Dimensions> * rightNode = nullptr;

		public:
		// Read-only traversal must not expose mutable descendants.
		const KdTreeNode * left_child() const { return leftNode; }
		const KdTreeNode * right_child() const { return rightNode; }

		/*void free_memory() -- current version does not recursevely delete child nodes
			{
			val.free_memory();
			vector<PointType>().swap(coords_max);
			vector<PointType>().swap(coords_min);
			delete leftNode;
			leftNode = nullptr;
			delete rightNode;
			rightNode = nullptr;
			}
			*/

		string output() const
			{
			ostringstream s1;
			s1 << "(";
			for (unsigned long il=0; il < val.coords.size(); il++)
				{
				s1 << val.coords[il];
				if (il< val.coords.size() - 1)
					s1 << ",";
				}
			s1 << "):" << val.index ;
			s1 << " min(" << kdtree_output_vector_as_string(coords_min, ",") << ") max(" <<  kdtree_output_vector_as_string(coords_max, ",") << ")";
			return(s1.str());
			}

		};

	template<size_t Dimensions>
	class KdTree
		{
		public:
		static_assert(Dimensions > 0, "A k-d tree must have at least one dimension.");
		static_assert(Dimensions <= static_cast<size_t>(numeric_limits<int32_t>::max()), "The serialized dimension must fit in int32_t.");
		using Point = Point_str<Dimensions>;
		using Node = KdTreeNode<Dimensions>;
		static constexpr int32_t dimension = static_cast<int32_t>(Dimensions);

		private:
		Node * root = nullptr;
		vector<Node> node_storage;

		// Validate external points at entry points, never per visited node.
		// Finite coordinates can still overflow distance arithmetic if too large.
		static void validate_point_coordinates(const Point &point)
			{
			for (PointType coordinate : point.coords)
				if (!std::isfinite(coordinate))
					throw std::domain_error("K-d tree coordinates must be finite.");
			}

		public:

		KdTree()
			{
			root = nullptr;
			}

    KdTree(const KdTree &) = delete;
    KdTree &operator=(const KdTree &) = delete;

		KdTree(vector<Point> &points) : KdTree()
			{
			buildKdTree(points);
			}

		~KdTree()
			{
			free_memory();
			}

		bool is_empty() const
			{
			if (root == nullptr)
				return true;
			return false;
			}

		bool buildKdTree(vector<Point> &points)
			{ //All the points must be of same dimension like (x, y ,z) or (x, y) or what ever you like
			if(points.empty() || points[0].coords.empty()) return false;

			if(points.size() > static_cast<size_t>(numeric_limits<int32_t>::max())) return false;

			// Reject invalid coordinates before changing either tree or input order.
			for (const Point &point : points)
				validate_point_coordinates(point);

			// Build separately: failures preserve the old tree and its node pointers.
			// The input vector can still be reordered during construction.
			KdTree replacement;
			replacement.node_storage.reserve(points.size());
			replacement.root = replacement.buildHelper(points, 0, points.size() - 1, 0);
			if (replacement.root == nullptr) return false;
			// Non-throwing commit; replacement now owns and destroys the old nodes.
			// Successful construction invalidates previous node pointers.
			node_storage.swap(replacement.node_storage);
			std::swap(root, replacement.root);
			return true;
			}


		private:
		Node * buildHelper(vector<Point>& points, const int32_t leftBorder, const int32_t rightBorder, const int32_t depth)
			{ //recursively create kd Tree
			if(leftBorder > rightBorder) return nullptr;
			int32_t curDim = depth % dimension;
			int32_t midInx = leftBorder + (rightBorder - leftBorder) / 2;
			//sort(points.begin() + leftBorder, points.begin() + rightBorder + 1, [curDim](const Point_str& a, const Point_str& b) { return a[curDim] < b[curDim]; });

			// ----------------------
			// ni nujno uporabiti celotnega sorta - namesto tega se lahko uporabi nth_element z �e nekaj dodatnega dela - to je potem kar hitrejse - upam da zadeva dela prav
			nth_element(points.begin() + leftBorder, points.begin() + midInx,  points.begin() + rightBorder + 1, [curDim](const Point& a, const Point& b) { return a.coords[curDim] < b.coords[curDim]; });
			// sedaj je treba �e na levi strani skupaj prestaviti vrednosti, ki so enake kot midInx (nth_element funkcija tega ne garantira)
			if (midInx - leftBorder > 1)
				{
				int32_t last_taken_indx=midInx;
				if (points[midInx-1].coords[curDim] == points[midInx].coords[curDim])
					last_taken_indx--;

				for (long il=midInx-2; il >= leftBorder; il--)
					if (points[il].coords[curDim] == points[midInx].coords[curDim])
						{
						swap(points[il],points[last_taken_indx-1]);
						last_taken_indx--;
						}
				}
			// ----------------------



			while(midInx > leftBorder && points[midInx - 1].coords[curDim] == points[midInx].coords[curDim]) // keep all the points with point[splitDim] >= midInx[splitDim] on the right of midInx
			{
				midInx -= 1;
			}

			/*cout << "depth: " << depth << endl;
			cout << "curDim: " << curDim << endl;
			for (int32_t il=leftBorder; il <= rightBorder; il++)
				{
				cout << points[il].output();
				if (il == midInx) cout << "*";
				cout << endl;
				}
			cout << "------" << endl;
			*/
			// node_storage was reserved for every input point before recursion,
			// so emplace_back cannot invalidate any node pointers during a build.
			node_storage.emplace_back(points[midInx]);
			Node * node = &node_storage.back();

			/*
			// calculate max and min coordinate values in the leaf for the Bounding Box data
			vector <PointType> min_coords = points[leftBorder].coords;
			vector <PointType> max_coords = points[leftBorder].coords;
			for (int32_t il=leftBorder + 1; il <= rightBorder; il++)
			for (int32_t ic=0; ic < dimension; ic ++)
				{
				if (points[il].coords[ic] < min_coords[ic]) min_coords[ic] = points[il].coords[ic];
				if (points[il].coords[ic] > max_coords[ic]) max_coords[ic] = points[il].coords[ic];
				}
			node->coords_max=max_coords;
			node->coords_min=min_coords;
			*/
			node->leftNode = buildHelper(points, leftBorder, midInx - 1, depth + 1);
			node->rightNode = buildHelper(points, midInx + 1, rightBorder, depth + 1);
			// Both children already have exact bounds; finish this node bottom-up.
			update_bounding_box(node);

			return node;
			}

		public:
		bool buildKdTree_and_do_not_change_the_kdtree_points_vector(const vector<Point> &kdtree_points)
			{
			vector<Point> kdtree_points_temp = kdtree_points;
			bool result = buildKdTree(kdtree_points_temp);
			vector<Point>().swap(kdtree_points_temp); // free memory
			return(result);
			}

		void free_memory()
			{
			root = nullptr;
			vector<Node>().swap(node_storage);
			}

		private:
		void release_node(Node * &node)
			{
			if (node == nullptr)
				return;
			node = nullptr;
			}

		public:
		void printKdTree2() const
			{
			if (root != nullptr)
				printKdTree2_helper(root, 0, 0, true);
			else
				cout << "Kdtree does not contain any nodes ! " << endl;
			}

		private:
		void printKdTree2_helper(const Node * const node, const int32_t depth, const int indent, const bool is_left) const
			{
			int delta_indent=12;
			for (int32_t il=0; il < (depth-1)*delta_indent; il++)
				cout << " ";
			if (depth > 0)
				for (int32_t il=0; il < delta_indent; il++)
					cout << "-";

			int32_t curDim = depth % dimension;

			cout << "L" << depth;
			if (depth > 0)
				{
				if (is_left) cout << "l";
				else cout << "r";
				}
			cout << ":";

			cout << "(";
			for (unsigned long il=0; il < node->val.coords.size(); il++)
				{
				if ((int32_t)il == curDim)
					cout << "*";
				cout << node->val.coords[il];
				if ( il < node->val.coords.size() - 1)
					cout << ",";
				}
			cout << "):" << node->val.index << " min(" << kdtree_output_vector_as_string(node->coords_min, ",") << ") max(" <<  kdtree_output_vector_as_string(node->coords_max, ",") << ")"<< endl;


			if(node->leftNode != nullptr)
				printKdTree2_helper(node->leftNode, depth +1, indent + delta_indent, true);
			else
				{
				for (int32_t il=0; il < (depth)*delta_indent; il++)
					cout << " ";
				for (int32_t il=0; il < delta_indent; il++)
					cout << "-";
				cout << "L" << depth+1 << ":""nullptr" << endl;
				}

			if(node->rightNode != nullptr)
				printKdTree2_helper(node->rightNode, depth +1, indent + delta_indent, false);
			else
				{
				for (int32_t il=0; il < (depth)*delta_indent; il++)
					cout << " ";
				for (int32_t il=0; il < delta_indent; il++)
					cout << "-";
				cout << "L" << depth+1 << ":""nullptr" << endl;
				}
			//indent = indent_old;
			}


		//print all Kd Tree's nodes layer by layer using breadth first search
		public:
		size_t count_number_of_nonnull_nodes() const
			{
			size_t counter=0;
			queue<Node *> q;
			q.emplace(root);
			Node * node;
			while(!q.empty())
				{
					node = q.front();
					q.pop();
					if(node != nullptr)
					{
						counter++;
						//cout << node->val.output() << endl;
						//cout << "(";
						//for_each(node->val.coords.begin(), node->val.coords.end() - 1, [](PointType& num) { cout << num << ", ";});
						//cout << *(node->val.coords.end() - 1) << ")" << endl;
						q.emplace(node->leftNode);
						q.emplace(node->rightNode);
					}
				}
			return(counter);
			}


		// rebuild a balanced kd-tree from scratch using all the points in the current tree - delanje tega med racunanjem PADa ni pohitrilo zadeve (2024-04)
		bool rebuild_a_balanced_KdTree()
			{
			// An empty tree is already balanced. No pointers are invalidated.
			if (root == nullptr) return true;
			vector<Point> points;
			// Preserve the original postorder without disconnecting any nodes.
			rebuild_a_balanced_KdTree_helper(root, points);
			// Direct construction also commits only after successful replacement.
			return buildKdTree(points);
			}

		private:
		void rebuild_a_balanced_KdTree_helper(const Node *node, vector<Point> &points) const
			{
			if (node == nullptr) return;

			if(node->leftNode != nullptr)
				rebuild_a_balanced_KdTree_helper(node->leftNode, points);
			if(node->rightNode != nullptr)
				rebuild_a_balanced_KdTree_helper(node->rightNode, points);
			// save this point
			points.push_back(node->val);
			}


		const Node * findMin(const Node * const node, const int32_t dim, const int32_t depth) const //find the node with the minimum value on dim dimension from depth
		{
			if(root == nullptr) return nullptr;
			const Node * minimum = node;
			findMinHelper(node, minimum, dim, depth);
			return minimum;
		}

		void findMinHelper(const Node * const node, const Node * &minimum, const int32_t dim, const int32_t depth) const
			{
			if(node == nullptr) return;
			if(node->val.coords[dim] < minimum->val.coords[dim]) { minimum = node; }
			// Skip subtrees whose lower bound cannot improve the current minimum.
			if (node->leftNode != nullptr && node->leftNode->coords_min[dim] < minimum->val.coords[dim])
				findMinHelper(node->leftNode, minimum, dim, depth + 1);
			if (depth % dimension != dim && node->rightNode != nullptr && node->rightNode->coords_min[dim] < minimum->val.coords[dim])
				findMinHelper(node->rightNode, minimum, dim, depth + 1);
			}

		void update_bounding_box(Node *node)
			{
			if (node == nullptr) return;

			node->coords_min = node->val.coords;
			node->coords_max = node->val.coords;

			if (node->leftNode != nullptr)
				{
				for (int32_t ic=0; ic < dimension; ic++)
					{
					node->coords_min[ic]=min(node->coords_min[ic],node->leftNode->coords_min[ic]);
					node->coords_max[ic]=max(node->coords_max[ic],node->leftNode->coords_max[ic]);
					}
				}

			if (node->rightNode != nullptr)
				{
				for (int32_t ic=0; ic < dimension; ic++)
					{
					node->coords_min[ic]=min(node->coords_min[ic],node->rightNode->coords_min[ic]);
					node->coords_max[ic]=max(node->coords_max[ic],node->rightNode->coords_max[ic]);
					}
				}
			}

		public:
		void deleteNode(const Point &point)
			{
			validate_point_coordinates(point);
			if(root == nullptr) return;
			if(deleteNodeHelper(root, point, 0))
				release_node(root);
			}

		private:
		bool deleteNodeHelper(Node * const node, const Point& point, const int32_t depth)
			{
			bool bounds_changed = false;
			return deleteNodeHelper(node, point, depth, bounds_changed);
			}

		bool deleteNodeHelper(Node * const node, const Point& point, const int32_t depth, bool &bounds_changed)
			{ //return true means the child node should be deleted
			bounds_changed = false;
			if(node == nullptr) return false;
			int32_t curDim = depth % dimension;
			if(point == node->val)
				{
				if(node->rightNode != nullptr)
					{
					const Node * minimumNode = findMin(node->rightNode, curDim, depth + 1);
					node->val = minimumNode->val; // do not swap(node->val, minimumNode) which would break the structure of kd-tree and result in not finding the node to delete
					if(deleteNodeHelper(node->rightNode, minimumNode->val, depth + 1, bounds_changed))
						release_node(node->rightNode);

					}
				else if(node->leftNode != nullptr)
					{
					const Node * minimumNode = findMin(node->leftNode, curDim, depth + 1);
					node->val = minimumNode->val;
					if(deleteNodeHelper(node->leftNode, minimumNode->val, depth + 1, bounds_changed))
						release_node(node->leftNode);

					if (node->rightNode != nullptr)
						release_node(node->rightNode);

					node->rightNode = node->leftNode;
					node->leftNode = nullptr;

					//cout << "bbb" << endl;
					}
				else
					{
					//cout << "ccc" << endl;
					// Removing a child always requires its parent to refresh its bounds.
					bounds_changed = true;
					return true; //return true to inform outter deleteNodeHelper to release this pointer;
					}
				}
			else
				{
				if(point.coords[curDim] < node->val.coords[curDim])
					{
					if(deleteNodeHelper(node->leftNode, point, depth + 1, bounds_changed))
						release_node(node->leftNode);
					}
				else
					{
					if(deleteNodeHelper(node->rightNode, point, depth + 1, bounds_changed))
						release_node(node->rightNode);
					}
				// This node's point and links are unchanged unless a child was removed.
				// If the child's bounds are also unchanged, this node's bounds stay exact.
				if (!bounds_changed) return false;
				}
			// Matching nodes always refresh: their point or child links changed,
			// even if deleting the replacement left the child's bounds unchanged.
			const auto previous_min = node->coords_min;
			const auto previous_max = node->coords_max;
			update_bounding_box(node);
			// Compare coordinate bytes so even signed-zero changes propagate exactly.
			bounds_changed = memcmp(previous_min.data(), node->coords_min.data(), sizeof(PointType) * Dimensions) != 0
				|| memcmp(previous_max.data(), node->coords_max.data(), sizeof(PointType) * Dimensions) != 0;
			return false;
			}


		public:
		const Node * getNode(const Point &point) const
			{
			validate_point_coordinates(point);
			if(root == nullptr) return nullptr;
			Node * node = root;
			int32_t depth = 0;
			int32_t curDim;
			while(node != nullptr)
			{
				if(node->val == point) return node;
				curDim = depth % dimension;
				if(node->val.coords[curDim] > point.coords[curDim])
				{
					node = node->leftNode;
				}
				else
				{
					node = node->rightNode;
				}
				depth += 1;
			}

			return nullptr;
			}


		private:
		vector<const Node *> get_all_SubNodes(const Node * const node) const
			{
			vector<const Node *> nodes;
			get_all_SubNodes_Helper(nodes, node);
			return(nodes);
			}

		void get_all_SubNodes_Helper(vector<const Node *> &nodes, const Node * const node) const
			{
			if (node != nullptr)
				{
				nodes.push_back(node);
				get_all_SubNodes_Helper(nodes,  node->leftNode);
				get_all_SubNodes_Helper(nodes,  node->rightNode);
				}
			}

		public:
		const Node * findNearestNode(const Point &point) const
			{
			validate_point_coordinates(point);
			if(root == nullptr) return nullptr;
			const Node * nearestNode = root;
			float minDist_sqr = calDist_sqr(point, nearestNode->val);
			findNearestNodeHelper<0>(root, point, minDist_sqr, nearestNode);
			return nearestNode;
			}

		// Preserve findNearestNode's float-distance ordering/ties, then apply an
		// inclusive radius check using double accumulation of coordinate products.
		// Thus this is nearest-then-filter, not a different tie-breaking policy.
		const Node * findNearestNode_in_radius(const Point &point, const double squared_euclidian_radius) const
			{
			validate_point_coordinates(point);
			KDTREE_ERRORIF(isnan(squared_euclidian_radius) || squared_euclidian_radius < 0);
			if (root == nullptr) return nullptr;

			// Inflate the initial bound conservatively for float accumulation and
			// strict '<' pruning, including zero/subnormal radii. The factor covers
			// rounding across Dimensions terms; double accumulation is more precise.
			const double rounding_budget = 2.0 * Dimensions * numeric_limits<float>::epsilon();
			const Node * nearestNode;
			if (!isfinite(squared_euclidian_radius) || rounding_budget >= 0.5 ||
				numeric_limits<PointType>::digits != numeric_limits<float>::digits)
				nearestNode = findNearestNode(point);
			else
				{
				const double expanded_radius =
					(squared_euclidian_radius + 2.0 * Dimensions * numeric_limits<float>::denorm_min()) /
					(1.0 - rounding_budget);
				if (expanded_radius >= numeric_limits<float>::max())
					nearestNode = findNearestNode(point);
				else
					{
					float minDist_sqr = nextafter(static_cast<float>(expanded_radius), numeric_limits<float>::infinity());
					const float root_distance = calDist_sqr(point, root->val);
					nearestNode = nullptr;
					// Keep the original root preference when it can affect the answer.
					if (root_distance < minDist_sqr)
						{
						nearestNode = root;
						minDist_sqr = root_distance;
						}
					findNearestNodeHelper<0>(root, point, minDist_sqr, nearestNode);
					}
				}

			if (nearestNode == nullptr) return nullptr;
			// Match PAD's arithmetic: do not replace these products with double
			// coordinate arithmetic or the tree's float-accumulated distance.
			double distance_sqr = 0;
			for (size_t axis = 0; axis < Dimensions; ++axis)
				distance_sqr += (point.coords[axis] - nearestNode->val.coords[axis]) *
					(point.coords[axis] - nearestNode->val.coords[axis]);
			return distance_sqr <= squared_euclidian_radius ? nearestNode : nullptr;
			}

		private:
		template<size_t Axis>
		void findNearestNodeHelper(const Node * const node, const Point& point, float& minDist_sqr, const Node * &nearestNode) const
			{
			static_assert(Axis < Dimensions, "The split axis must be within the tree dimensions.");
			// Cycle through compile-time axes without storing an axis in each node.
			constexpr size_t nextAxis = (Axis + 1) % Dimensions;
			if(node == nullptr) return;
			//float dist = calDist(point, node->val);
			float dist_sqr = calDist_sqr(point, node->val);
			if(dist_sqr < minDist_sqr)
				{
				nearestNode = node;
				minDist_sqr = dist_sqr;
				}
			// Visit the child with the smaller bounding-box lower bound first.
			// Equal bounds retain split-side preference, but unequal bounds can
			// change equal-distance candidate selection relative to the old search.
			const bool left_first = point.coords[Axis] < node->val.coords[Axis];
			const Node *first = left_first ? node->leftNode : node->rightNode;
			const Node *second = left_first ? node->rightNode : node->leftNode;
			float first_bound = first ? distance_to_node_bounds_sqr(point, first) : numeric_limits<float>::infinity();
			float second_bound = second ? distance_to_node_bounds_sqr(point, second) : numeric_limits<float>::infinity();
			if (second_bound < first_bound)
				{
				std::swap(first, second);
				std::swap(first_bound, second_bound);
				}
			if (first != nullptr && first_bound < minDist_sqr)
				findNearestNodeHelper<nextAxis>(first, point, minDist_sqr, nearestNode);
			// Reuse the bound, but compare against the improved search radius.
			if (second != nullptr && second_bound < minDist_sqr)
				findNearestNodeHelper<nextAxis>(second, point, minDist_sqr, nearestNode);

			}

		float distance_to_node_bounds_sqr(const Point &point, const Node *node) const
			{
			float sum = 0;
			for (size_t axis = 0; axis < Dimensions; ++axis)
				{
				const float delta = point.coords[axis] - clip_value_to_min_max(
					point.coords[axis], node->coords_min[axis], node->coords_max[axis]);
				sum += delta * delta;
				}
			return sum;
			}

		float calDist_sqr(const Point& a, const Point& b) const
			{
			float sum = 0;
			for(int32_t inx = 0; inx < dimension; inx++)
				{
				float temp = (float)(b.coords[inx] - a.coords[inx]);
				sum += temp*temp;
				}
			return(sum);
			}

		bool test_if_the_hypersphere_and_hyperrectangle_intersect(const array<PointType, Dimensions> &circle_center, const float radius, const array<PointType, Dimensions> &rect_coords_min, const array<PointType, Dimensions> &rect_coords_max) const
			{
			/*
			// get the point of rectangle that is nearest the circle center
			vector <PointType> nearest_point;
			for (int32_t ic=0; ic < dimension; ic++)
				nearest_point.push_back(clip_value_to_min_max(circle_center[ic],rect_coords_min[ic], rect_coords_max[ic]));

			// the distance between the nearest point and circle center
			float sqr_distance = squared_euclidian_distanceX(circle_center, nearest_point);
			*/

			float sqr_distance = 0;
			for (int32_t ic=0; ic < dimension; ic++)
				{
				float temp = circle_center[ic] - clip_value_to_min_max(circle_center[ic],rect_coords_min[ic], rect_coords_max[ic]);
				sqr_distance+= temp*temp;
				}

			//cout << "- (" <<  kdtree_output_vector_as_string(nearest_point, ",") << ") " << sqrt(sqr_distance) << endl;

			if (sqr_distance < radius * radius)
				return(true);
			return(false);
			}

		// Nearest-neighbor pruning stays strict; cluster searches include tangency.
		template<bool IncludeBoundary = false>
		bool test_if_the_hypersphere_and_hyperrectangle_intersect_using_sqr_radius(const array<PointType, Dimensions> &circle_center, const float sqr_radius, const array<PointType, Dimensions> &rect_coords_min, const array<PointType, Dimensions> &rect_coords_max) const
			{
			/*
			// get the point of rectangle that is nearest the circle center
			vector <PointType> nearest_point;
			for (int32_t ic=0; ic < dimension; ic++)
				nearest_point.push_back(clip_value_to_min_max(circle_center[ic],rect_coords_min[ic], rect_coords_max[ic]));

			// the distance between the nearest point and circle center
			float sqr_distance = squared_euclidian_distanceX(circle_center, nearest_point);
			*/

			float sqr_distance = 0;
			for (int32_t ic=0; ic < dimension; ic++)
				{
				float temp = circle_center[ic] - clip_value_to_min_max(circle_center[ic],rect_coords_min[ic], rect_coords_max[ic]);
				/*float temp = 0;
				float cc = circle_center[ic];
				if ( cc < rect_coords_min[ic]) temp= cc - rect_coords_min[ic];
				else if (cc > rect_coords_max[ic]) temp= cc - rect_coords_max[ic];*/
				sqr_distance+= temp*temp;
				}

			//cout << "- (" <<  kdtree_output_vector_as_string(nearest_point, ",") << ") " << sqrt(sqr_distance) << endl;

			return IncludeBoundary ? sqr_distance <= sqr_radius : sqr_distance < sqr_radius;
			}


		bool test_if_the_hyperrectangle_is_fully_inside_the_sphere(const array<PointType, Dimensions> &circle_center, const float radius, const array<PointType, Dimensions> &rect_coords_min, const array<PointType, Dimensions> &rect_coords_max) const
			{
			float sqr_distance = 0;
			for (int32_t ic=0; ic < dimension; ic++)
				{
				float temp = max(fabs(circle_center[ic] - rect_coords_min[ic]), fabs(circle_center[ic] - rect_coords_max[ic]));
				sqr_distance+= temp*temp;
				}

			if (sqr_distance < radius * radius)
				return(true);
			return(false);
			}

		bool test_if_the_hyperrectangle_is_fully_inside_the_sphere_using_sqr_radius(const array<PointType, Dimensions> &circle_center, const float sqr_radius, const array<PointType, Dimensions> &rect_coords_min, const array<PointType, Dimensions> &rect_coords_max) const
			{
			float sqr_distance = 0;
			for (int32_t ic=0; ic < dimension; ic++)
				{
				float temp = max(fabs(circle_center[ic] - rect_coords_min[ic]), fabs(circle_center[ic] - rect_coords_max[ic]));
				sqr_distance+= temp*temp;
				}

			if (sqr_distance < sqr_radius)
				return(true);
			return(false);
			}


		PointType clip_value_to_min_max (const PointType &val, const PointType &min_limit, const PointType &max_limit) const
			{
			if (val < min_limit) return min_limit;
			if (val > max_limit) return max_limit;
			return(val);
 			}


		public:
		vector<const Node *> findNearestNodeCluster(const Point& point, const float distance) const
			{ //find all nodes with the distance from which to the "point" is less than or equal to the "distance."
			validate_point_coordinates(point);
			// Positive infinity remains allowed; reject invalid radii once per call.
			if (std::isnan(distance) || distance < 0)
				throw std::domain_error("Cluster radius must be nonnegative and not NaN.");
			vector<const Node *> cluster;
			if(root == nullptr) return cluster;
			float distance_squared = distance*distance;
			findNearestNodeClusterHelper(root, point, distance, distance_squared, cluster, 0);
			return cluster;
			}


		private:
		void findNearestNodeClusterHelper(const Node * const node, const Point& point, const float distance, const float distance_sqaured, vector<const Node *> &cluster, const int32_t depth) const
			{
			if(node == nullptr) return;

			/*
			if (test_if_the_hyperrectangle_is_fully_inside_the_sphere( point.coords, distance, node->coords_min, node->coords_max))
				{
				vector<std::shared_ptr<KdTreeNode>> subnodes = get_all_SubNodes(node);
				cluster.insert(cluster.end(), subnodes.begin(), subnodes.end());
				}

			else if (test_if_the_hypersphere_and_hyperrectangle_intersect( point.coords, distance, node->coords_min, node->coords_max ))
				{
				float dist_sqr = calDist_sqr(point, node->val);
				if(dist_sqr < distance_sqaured)
					cluster.emplace_back(node);

				findNearestNodeClusterHelper(node->rightNode, point, distance, distance_sqaured, cluster, depth + 1);
				findNearestNodeClusterHelper(node->leftNode, point, distance, distance_sqaured, cluster, depth + 1);
				}
			*/

			/*
			if (test_if_the_hypersphere_and_hyperrectangle_intersect( point.coords, distance, node->coords_min, node->coords_max ))
				{

				if (test_if_the_hyperrectangle_is_fully_inside_the_sphere( point.coords, distance, node->coords_min, node->coords_max))
					{
					vector<std::shared_ptr<KdTreeNode>> subnodes = get_all_SubNodes(node);
					cluster.insert(cluster.end(), subnodes.begin(), subnodes.end());
					}

				else
					{
					float dist_sqr = calDist_sqr(point, node->val);
					if(dist_sqr < distance_sqaured)
						cluster.emplace_back(node);

					findNearestNodeClusterHelper(node->rightNode, point, distance, distance_sqaured, cluster, depth + 1);
					findNearestNodeClusterHelper(node->leftNode, point, distance, distance_sqaured, cluster, depth + 1);
					}
				}

			*/



			if (test_if_the_hypersphere_and_hyperrectangle_intersect_using_sqr_radius<true>( point.coords, distance_sqaured, node->coords_min, node->coords_max ))
				{

				float dist_sqr = calDist_sqr(point, node->val);
				bool inside_sphere = false;
				if(dist_sqr <= distance_sqaured)
					inside_sphere = true;

				if (inside_sphere && test_if_the_hyperrectangle_is_fully_inside_the_sphere_using_sqr_radius( point.coords, distance_sqaured, node->coords_min, node->coords_max))
					{
					get_all_SubNodes_Helper(cluster, node);
					}

				else
					{
					if (inside_sphere)
						cluster.emplace_back(node);

					findNearestNodeClusterHelper(node->rightNode, point, distance, distance_sqaured, cluster, depth + 1);
					findNearestNodeClusterHelper(node->leftNode, point, distance, distance_sqaured, cluster, depth + 1);
					}
				}


			/*
			if (test_if_the_hypersphere_and_hyperrectangle_intersect( point.coords, distance, node->coords_min, node->coords_max ))
				{
				float dist_sqr = calDist_sqr(point, node->val);
				if(dist_sqr < distance_sqaured)
					cluster.emplace_back(node);

				findNearestNodeClusterHelper(node->rightNode, point, distance, distance_sqaured, cluster, depth + 1);
				findNearestNodeClusterHelper(node->leftNode, point, distance, distance_sqaured, cluster, depth + 1);
				}

			*/


			/*
			int32_t curDim = depth % dimension;
			float dist_sqr = calDist_sqr(point, node->val);
			if(dist_sqr < distance_sqaured)
				cluster.emplace_back(node);

			float delta = point.coords[curDim] - node->val.coords[curDim];
			if(delta < 0)
				{
				findNearestNodeClusterHelper(node->leftNode, point, distance, distance_sqaured, cluster, depth + 1);
				if( -delta <= distance)
					//if (test_if_the_hypersphere_and_hyperrectangle_intersect( point.coords, distance, node->coords_min, node->coords_max ))
						{
						findNearestNodeClusterHelper(node->rightNode, point, distance, distance_sqaured, cluster, depth + 1);
						}
	 			}
			else
				{
				findNearestNodeClusterHelper(node->rightNode, point, distance, distance_sqaured, cluster, depth + 1);
				if(delta < distance)
					//if (test_if_the_hypersphere_and_hyperrectangle_intersect( point.coords, distance, node->coords_min, node->coords_max ))
						{
						findNearestNodeClusterHelper(node->leftNode, point, distance, distance_sqaured, cluster, depth + 1);
						}
				}


			*/
			}



        private:
        /*
         * K-d tree serialization (version 1)
         *
         * Serialization and file I/O are independent:
         *   auto bytes = tree.serialize();          // No disk access.
         *   other_tree.deserialize(bytes);         // No disk access.
         *   kdtree_binary_io::write_file("tree.bin", bytes);
         *   auto restored_bytes = kdtree_binary_io::read_file("tree.bin");
         *   other_tree.deserialize(restored_bytes);
         *
         * deserialize(const uint8_t*, size_t) accepts a borrowed buffer; the
         * caller must supply accessible storage of that size. No buffer is
         * retained. serialize() returns an owning vector; no manual delete[] is
         * needed. Disk methods wrap these in-memory APIs and byte-file helpers.
         * Successful deserialization replaces the tree and invalidates previous
         * node pointers. Failure leaves the current tree intact. Empty trees are
         * supported. Search, deletion, and balancing policies are unchanged.
         *
         * Format: all multibyte values are little-endian. Coordinates are
         * IEEE-754 bit patterns, not raw C++ structs; there is no struct padding.
         * The 40-byte header is:
         *   Offset  Bytes  Value
         *        0      8  ASCII PADKDTRE
         *        8      4  Version (1)
         *       12      4  Dimension count
         *       16      4  Coordinate width (4 or 8 bytes)
         *       20      4  Unsigned point-index width (1, 2, 4 or 8 bytes)
         *       24      8  Reachable node count
         *       32      8  Root record index (0; UINT64_MAX for an empty tree)
         * Each record contains Dimensions coordinates, one original point
         * index, a u64 left-child record index, and a u64 right-child record
         * index. UINT64_MAX denotes a missing child. Record size is
         * Dimensions * sizeof(PointType) + sizeof(IndexType) + 16.
         * Child record indices are distinct from original point indices.
         *
         * The writer assigns breadth-first indices to reachable nodes only;
         * deleted storage slots are omitted. Every child follows its parent.
         * Loading preserves exact coordinates, indices, and left/right topology
         * without rebalancing. Bounds are omitted and recomputed bottom-up,
         * combining node, left subtree, and right subtree as in construction.
         *
         * Loading requires matching dimensions/type widths, exact buffer length,
         * finite coordinates, forward-only child links, a unique parent for each
         * non-root node, full connectivity, and valid split partitions
         * (left < split; right >= split). The node-count limit is INT32_MAX,
         * as in construction. Parsing and bounds reconstruction are iterative.
         * There is no checksum: structurally valid corruption may go undetected.
         * Old files are incompatible and must be regenerated. Changing PointType
         * or IndexType widths requires a matching reader; no conversion is done.
         *
         * File helpers use automatic cleanup, chunked transfers, and explicit
         * checks for opening, sizing/seeking, reading/writing, flushing, and
         * closing. They do not interpret tree records. Writing truncates an
         * existing file; it is not atomic and does not promise crash durability.
         * Use a separate path when the old file must remain recoverable.
         *
         * Verification: statically reviewed only, not compiled or executed.
         * Runtime checks still needed: round trips for empty trees, single nodes,
         * duplicate coordinates, and partially deleted trees; identical queries
         * before/after loading; rejection of truncated, oversized, wrong-version,
         * wrong-type, cyclic, shared-child, disconnected, non-finite, and
         * split-invalid records without changing the destination tree.
         */
        static void append_unsigned(vector<uint8_t> &bytes, uint64_t value, size_t width)
            {
            for (size_t i = 0; i < width; ++i)
                bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
            }

        static uint64_t read_unsigned(const uint8_t *bytes, size_t size, size_t &position, size_t width)
            {
            KDTREE_ERRORIF(width > 8 || position > size || width > size - position);
            uint64_t value = 0;
            for (size_t i = 0; i < width; ++i)
                value |= static_cast<uint64_t>(bytes[position++]) << (8 * i);
            return value;
            }

        static void check_binary_types()
            {
            static_assert(CHAR_BIT == 8, "Binary format requires 8-bit bytes.");
            static_assert(std::is_floating_point<PointType>::value &&
                numeric_limits<PointType>::is_iec559 &&
                (sizeof(PointType) == 4 || sizeof(PointType) == 8),
                "Binary coordinates require IEEE-754 float32 or float64.");
            static_assert(std::is_integral<IndexType>::value && std::is_unsigned<IndexType>::value &&
                (sizeof(IndexType) == 1 || sizeof(IndexType) == 2 ||
                 sizeof(IndexType) == 4 || sizeof(IndexType) == 8),
                "Binary point indices require an unsigned 8/16/32/64-bit integer.");
            }

        static size_t binary_record_size()
            {
            const size_t fixed_bytes = sizeof(IndexType) + 16;
            KDTREE_ERRORIF(Dimensions > (numeric_limits<size_t>::max() - fixed_bytes) / sizeof(PointType));
            return Dimensions * sizeof(PointType) + fixed_bytes;
            }

        public:
        // Pure in-memory serialization. No disk access, no raw pointer values.
        vector<uint8_t> serialize() const
            {
            check_binary_types();
            const uint64_t missing = numeric_limits<uint64_t>::max();
            vector<const Node *> nodes;
            // Also validates that child pointers belong to this tree's storage.
            unordered_map<const Node *, uint64_t> ids;
            if (root != nullptr)
                {
                ids.reserve(node_storage.size());
                for (const Node &node : node_storage) ids.emplace(&node, missing);
                auto root_entry = ids.find(root);
                KDTREE_ERRORIF(root_entry == ids.end());
                root_entry->second = 0;
                nodes.push_back(root);
                // Breadth-first indexing preserves left/right links, not a rebuild.
                for (size_t i = 0; i < nodes.size(); ++i)
                    {
                    const Node *children[2] = {nodes[i]->leftNode, nodes[i]->rightNode};
                    for (const Node *child : children)
                        if (child != nullptr)
                            {
                            auto entry = ids.find(child);
                            KDTREE_ERRORIF(entry == ids.end());
                            // A second incoming edge also catches cycles.
                            KDTREE_ERRORIF(entry->second != missing);
                            entry->second = static_cast<uint64_t>(nodes.size());
                            nodes.push_back(child);
                            }
                    }
                }
            KDTREE_ERRORIF(nodes.size() > static_cast<size_t>(numeric_limits<int32_t>::max()));
            const size_t record_size = binary_record_size();
            vector<uint8_t> bytes;
            KDTREE_ERRORIF(bytes.max_size() < 40 || nodes.size() > (bytes.max_size() - 40) / record_size);
            bytes.reserve(40 + nodes.size() * record_size);
            const uint8_t magic[8] = {'P','A','D','K','D','T','R','E'};
            bytes.insert(bytes.end(), magic, magic + 8);
            append_unsigned(bytes, 1, 4);
            append_unsigned(bytes, Dimensions, 4);
            append_unsigned(bytes, sizeof(PointType), 4);
            append_unsigned(bytes, sizeof(IndexType), 4);
            append_unsigned(bytes, nodes.size(), 8);
            append_unsigned(bytes, nodes.empty() ? missing : 0, 8);
            using CoordinateBits = typename std::conditional<sizeof(PointType) == 4, uint32_t, uint64_t>::type;
            for (const Node *node : nodes)
                {
                for (PointType coordinate : node->val.coords)
                    {
                    KDTREE_ERRORIF(!isfinite(coordinate));
                    CoordinateBits bits = 0;
                    memcpy(&bits, &coordinate, sizeof(coordinate));
                    append_unsigned(bytes, bits, sizeof(coordinate));
                    }
                append_unsigned(bytes, node->val.index, sizeof(IndexType));
                append_unsigned(bytes, node->leftNode ? ids.at(node->leftNode) : missing, 8);
                append_unsigned(bytes, node->rightNode ? ids.at(node->rightNode) : missing, 8);
                }
            return bytes;
            }

        // Pure in-memory load. Validate temporary storage before committing, so
        // an exception leaves the current tree intact. No recursive parsing.
        void deserialize(const uint8_t *bytes, const size_t size)
            {
            check_binary_types();
            const uint64_t missing = numeric_limits<uint64_t>::max();
            const uint8_t magic[8] = {'P','A','D','K','D','T','R','E'};
            KDTREE_ERRORIF(bytes == nullptr || size < 40);
            KDTREE_ERRORIF(memcmp(bytes, magic, 8) != 0);
            size_t position = 8;
            KDTREE_ERRORIF(read_unsigned(bytes, size, position, 4) != 1);
            KDTREE_ERRORIF(read_unsigned(bytes, size, position, 4) != Dimensions);
            KDTREE_ERRORIF(read_unsigned(bytes, size, position, 4) != sizeof(PointType));
            KDTREE_ERRORIF(read_unsigned(bytes, size, position, 4) != sizeof(IndexType));
            const uint64_t count = read_unsigned(bytes, size, position, 8);
            const uint64_t root_id = read_unsigned(bytes, size, position, 8);
            KDTREE_ERRORIF(count > static_cast<uint64_t>(numeric_limits<int32_t>::max()));
            KDTREE_ERRORIF((count == 0 && root_id != missing) || (count != 0 && root_id != 0));
            const size_t record_size = binary_record_size();
            const size_t payload = size - position;
            KDTREE_ERRORIF(payload % record_size != 0 || count != payload / record_size);

            vector<Node> loaded;
            KDTREE_ERRORIF(count > loaded.max_size());
            loaded.resize(static_cast<size_t>(count)); // Addresses remain stable.
            vector<uint8_t> incoming(loaded.size(), 0);
            vector<size_t> axes(loaded.size(), 0);
            using CoordinateBits = typename std::conditional<sizeof(PointType) == 4, uint32_t, uint64_t>::type;
            for (size_t i = 0; i < loaded.size(); ++i)
                {
                Node &node = loaded[i];
                for (PointType &coordinate : node.val.coords)
                    {
                    const CoordinateBits bits = static_cast<CoordinateBits>(
                        read_unsigned(bytes, size, position, sizeof(PointType)));
                    memcpy(&coordinate, &bits, sizeof(coordinate));
                    KDTREE_ERRORIF(!isfinite(coordinate));
                    }
                node.val.index = static_cast<IndexType>(read_unsigned(bytes, size, position, sizeof(IndexType)));
                const uint64_t children[2] = {
                    read_unsigned(bytes, size, position, 8),
                    read_unsigned(bytes, size, position, 8)};
                for (size_t side = 0; side < 2; ++side)
                    {
                    const uint64_t child = children[side];
                    if (child == missing) continue;
                    // Parent-before-child indices rule out cycles and root links.
                    KDTREE_ERRORIF(child >= count || child <= i);
                    const size_t child_index = static_cast<size_t>(child);
                    KDTREE_ERRORIF(incoming[child_index] != 0);
                    incoming[child_index] = 1;
                    axes[child_index] = (axes[i] + 1) % Dimensions;
                    if (side == 0) node.leftNode = &loaded[child_index];
                    else node.rightNode = &loaded[child_index];
                    }
                }
            KDTREE_ERRORIF(position != size);
            // Together with forward-only edges, this proves connectivity to root.
            for (size_t i = 1; i < loaded.size(); ++i)
                KDTREE_ERRORIF(incoming[i] != 1);

            // Recreate exact bounds in node/left/right order, as construction does.
            for (size_t i = loaded.size(); i > 0; --i)
                {
                Node &node = loaded[i - 1];
                update_bounding_box(&node);
                const size_t axis = axes[i - 1];
                // Validate the split invariant for the entire child subtrees.
                if (node.leftNode)
                    KDTREE_ERRORIF(!(node.leftNode->coords_max[axis] < node.val.coords[axis]));
                if (node.rightNode)
                    KDTREE_ERRORIF(!(node.rightNode->coords_min[axis] >= node.val.coords[axis]));
                }
            node_storage.swap(loaded);
            root = node_storage.empty() ? nullptr : &node_storage[0];
            }

        void deserialize(const vector<uint8_t> &bytes)
            {
            deserialize(bytes.data(), bytes.size());
            }

        // Thin convenience wrappers; byte encoding and file transport stay separate.
        void write_KdTree_to_a_binary_file(const string fname) const
            {
            kdtree_binary_io::write_file(fname, serialize());
            }

        void read_KdTree_from_a_binary_file(const string fname)
            {
            const vector<uint8_t> bytes = kdtree_binary_io::read_file(fname);
            deserialize(bytes);
            }

		};

}

#undef KDTREE_ERRORIF
