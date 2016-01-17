#pragma once

struct Size{
	Size() = default;
	Size(float w, float h) :width(w), height(h) {}
	float width, height;
};