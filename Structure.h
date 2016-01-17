#pragma once

#include<vector>

namespace mtl{

struct Size{
	Size() = default;
	Size(float w, float h) :width(w), height(h) {}
	float width, height;
};

typedef std::vector< std::vector<float> > Map;

}