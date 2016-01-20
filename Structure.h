#pragma once

#include<array>

namespace mtl{

template< std::size_t SIZE >
using Map = std::array< std::array<float, SIZE>, SIZE >;

struct Size {
	float height, width;
	Size(float w, float h) :width(w), height(h) {}
	Size() = default;
};

struct Range {
	float min_, max_;
	constexpr Range(float _min, float _max) :min_(_min), max_(_max) {}
	Range() = default;
};
	
struct Point {
	float x, y;
	Point(float _x, float _y) :x(_x), y(_y) {}
	Point() = default;
};

}