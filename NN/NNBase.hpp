//
//  NNBase.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/13.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __MTL_Development__NNBase__
#define __MTL_Development__NNBase__

#include<cmath>
#include<amp.h>
#include<amp_math.h>
#include <array>
#include <utility>
#include <fstream>
#include "Utility.hpp"
#include "../Structure.h"
#include "../configuration.h"

LIB_SCOPE_BEGIN()

template<std::size_t WEIGHT_SIZE>
class FeedForward_Amp;

template<class NetworkStruct,class ActivationObject>
struct calcConvolution;

template<class NetworkStruct, class ActivationObject, class Tag>
struct _calcSurface;

template<class NetworkStruct, class ActivationObject>
using calcSurface = _calcSurface < NetworkStruct, ActivationObject, typename NetworkStruct::tag >;

template<class T>
struct ActivationFunc{
    ActivationFunc(){
        static_assert(std::is_same<decltype(std::declval<T>().activate(std::declval<double>())),double>::value,"activate is not defined");
        static_assert(std::is_same<decltype(std::declval<T>().activateDerivative(std::declval<double>())),double>::value,"activateDerivative is not defined");
    }
};

template<std::size_t _NEXT_LAYER_SIZE>
class Unit{
public:
    double bias=0.5;
    std::array<float,_NEXT_LAYER_SIZE> weight;
    
    template<class F>
    double output(F&& f);
    
    template<std::size_t PREV_LAYER_SIZE,std::size_t _iSize>
    double input(const std::array<Unit<PREV_LAYER_SIZE> , _iSize>& surface);
    
    void    setStatus(double _s){_status = _s;}
    double  getStatus()const{return _status;}
private:
    float _status;
};
template<std::size_t _NEXT_LAYER_SIZE>
template<class F>
double Unit<_NEXT_LAYER_SIZE>::output(F&& f){
    return f(_status+bias);
}

class Unit_Dy{
public:
	typedef float Status_t;
    double bias=0.5;
    std::vector<double> weight;
    
    template<class F>
    double output(F&& f)const;
    
    template<std::size_t _iSize>
    double input(const std::array<Unit_Dy , _iSize>& surface);
    
    void    setStatus(double _s){_status = _s;}
    double  getStatus()const{return _status;}
private:
    float _status;
};

template<class F>
double Unit_Dy::output(F&& f)const{
    return f(_status+bias);
}

class Unit_Dy_Litteral {
public:
	float bias = 0.5;
	float* weight;

	template<class F>
	double output(F&& f);

	void    setStatus(double _s){ _status = _s; }
	double  getStatus()const{ return _status; }
private:
	float _status;
};

template<class F>
double Unit_Dy_Litteral::output(F&& f) {
	return f(_status + bias);
}

template<std::size_t WEIGHT_SIZE>
class Unit_Dy_Amp{
	friend FeedForward_Amp<WEIGHT_SIZE>;
public:
	static constexpr int W_SIZE = WEIGHT_SIZE;
	float bias = 0.5;
	float weight[W_SIZE];

	template<class F>
	float output(F&& f)const restrict(cpu);

	template<class F>
	float output_amp()const restrict(amp);

	void   setStatus(float _s) restrict(cpu,amp){ _status = _s; }
	float  getStatus()const restrict(cpu,amp) { return _status; }
private:
	float _status;
};

template<std::size_t WEIGHT_SIZE>
template<class F>
float Unit_Dy_Amp<WEIGHT_SIZE>::output(F&& f)const restrict(cpu){
	return f(_status + bias);
}

template<std::size_t WEIGHT_SIZE>
template<class F>
float Unit_Dy_Amp<WEIGHT_SIZE>::output_amp()const restrict(amp){
	F f;
	return f(_status + bias);
}

template<std::size_t FilterSize>
class Map_Amp {
public:
	typedef std::vector<float> Status_t;
	void setStatus(float x, float y,float _s){ map[y*size.width +x] = _s; }
	void setStatus(std::vector<float> v) { map = v; }
	
	/*float  getStatus(float x, float y)const restrict(cpu, amp) {
		using c_i = concurrency::index<2>;
		concurrency::index<2> idx;
		return map_view[c_i(y-1,x-1)] + map_view[c_i(y - 1,x)] + map_view[c_i(y - 1,x+1)] + map_view[c_i(y,x - 1)] + map_view[c_i(y,x)] + map_view[c_i(y,x+1)] + map_view[c_i(y + 1,x+1)] + map_view[c_i(y + 1,x)] + map_view[c_i(y + 1,x+1)];
	}*/
	float getStatus(float x,float y)const{
		return map[y*size.width + x];
	}

	template<class F>
	float output(F&& f)const;

	void setSizeOfMap(Size _size);
	std::vector< Map<FilterSize> > weight;
	std::vector< float > map;
	float bias;
	Size size;

	std::vector<float>& getMap() { return map; }
	/*concurrency::array_view< float, 2 > map_view;
	concurrency::array_view< float, 2 > bias_view;
	concurrency::array_view< float, 2 > w_mat_view;*/
private:
	
};

template<std::size_t FilterSize>
void Map_Amp<FilterSize>::setSizeOfMap(Size _size) {
	map.resize(size.width * size.height);
	size = _size;
}
template<std::size_t FilterSize>
template<class F>
float Map_Amp<FilterSize>::output(F&& f)const{
	return std::accumulate(map.begin(), map.end(), 0.0f);
}

/* Feed forward perceptron class */
/* This is the Perceptron model that does not have the recursive structure. */
/* This is implemented by std::tuple. Therefore, this require tuple mtl-utility(See Utility.hpp).  */

struct STATIC{};
struct DYNAMIC{};
struct AMP {};
struct COMPOSITE {};

template<std::size_t _First , std::size_t _Last , std::size_t... Args>
class FeedForward{
public:
	//Structure of network determined at the time of compilation
    typedef typename mtl::make_tuple_array_3dims<Unit, std::tuple<>, _First, Args..., _Last>::type structure;
	typedef STATIC tag;

	static constexpr std::size_t LAYER_SIZE = std::tuple_size<structure>::value;
    
    structure network;
    
    template<std::size_t layer_index>
    using layer_type = typename std::tuple_element<layer_index,structure>::type;
    
    template<std::size_t layer_index>
    static constexpr std::size_t getLayerSize(){return std::tuple_size< typename std::tuple_element<layer_index,structure>::type >::value;}
    
    template<std::size_t layer_index>
    typename std::tuple_element<layer_index, structure>::type& getLayer(){return std::get<layer_index>(network);};
    
    template<std::size_t layer_index,std::size_t unit_index>
    mtl::array_base_t<typename std::tuple_element<layer_index, structure>::type>& getUnit(){return std::get<unit_index>(std::get<layer_index>(network));};
   
    template<std::size_t layer_index,std::size_t unit_index>
    typename std::tuple_element<layer_index+1, structure>::type& layerForwardIterator(); //forward iterator for propagation.
    
    template<std::size_t layer_index,std::size_t unit_index>
    typename std::tuple_element<layer_index-1, structure>::type& layerBackwordIterator(); //backward iterator for propagation.
};

template<std::size_t _First , std::size_t _Last , std::size_t... Args>
template<std::size_t layer_index,std::size_t unit_index>
auto FeedForward<_First,_Last,Args...>::layerForwardIterator()
->typename std::tuple_element<layer_index+1, structure>::type&{
    static_assert(layer_index+1 < LAYER_SIZE,"OUT OF RANGE");
    return std::get<layer_index+1>(network);
}

template<std::size_t _First , std::size_t _Last , std::size_t... Args>
template<std::size_t layer_index,std::size_t unit_index>
auto FeedForward<_First,_Last,Args...>::layerBackwordIterator()
->typename std::tuple_element<layer_index-1, structure>::type&{
    static_assert((long)layer_index-1 >= 0,"OUT OF RANGE");
    return std::get<layer_index-1>(network);
}


class FeedForward_Dy{
public:
	//Structure of network determined at the runtime
    typedef std::vector<std::vector<Unit_Dy>> structure;
	typedef Unit_Dy Unit_t;
	typedef DYNAMIC tag;
	typedef std::vector<size_t> struct_t;
	//C++ AMP-Restricted Function is not allow std::size_t(unsigned long)
	typedef unsigned int size_t;

	template<class ActivationObject>
	using Calc_Func = calcSurface<FeedForward_Dy, ActivationObject>;

	FeedForward_Dy(const struct_t&, Range);
	FeedForward_Dy() = default;

    structure network;
	
	size_t getNumberOfLayers(){ return static_cast<size_t>(network.size()); }
    size_t getNumberOfUnits(size_t layer_index){return static_cast<size_t>(network[layer_index].size());}
    
    std::vector<Unit_Dy>& getLayer(size_t layer_index){return network[layer_index];};
    
    Unit_Dy& getUnit(size_t layer_index, size_t unit_index){return network[layer_index][unit_index];};
    
    void setNumberOfLayers(size_t size) { network.resize(size); }
    void setNumberOfUnits(size_t layer_index, size_t size) { network[layer_index].resize(size); };
   
    std::vector<Unit_Dy>& layerForwardIterator(size_t layer_index,size_t unit_index); //forward iterator for propagation.
    std::vector<Unit_Dy>& layerBackwordIterator(size_t layer_index,size_t unit_index);//backward iterator for propagation.
    
    bool exportNetwork(std::string filename);
    bool importNetwork(std::string filename);
};

FeedForward_Dy::FeedForward_Dy(const struct_t& number_of_units, Range range) {
	setNumberOfLayers(static_cast<size_t>(number_of_units.size()));
	for (size_t i = 0; i<number_of_units.size(); i++) {
		setNumberOfUnits(i, number_of_units[i]);
	}

	for (size_t i = 0; i<number_of_units.size() - 1; i++) {
		//todo autholize
		for (size_t j = 0; j<number_of_units[i]; j++) {
			network[i][j].weight.resize(number_of_units[i + 1]);
		}
	}

	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<double> distribution(range.min_,range.max_);

	for (size_t i = 0; i<getNumberOfLayers(); i++) {
		for (size_t j = 0; j<getNumberOfUnits(i); j++) {
			network[i][j].bias = 0;
			if (i == getNumberOfLayers() - 1) {
				std::fill(network[i][j].weight.begin(), network[i][j].weight.end(), 0.0f);
			}
			else std::fill(network[i][j].weight.begin(), network[i][j].weight.end(), distribution(mt));
		}
	}
}

bool FeedForward_Dy::exportNetwork(std::string filename){
    
    std::ofstream ofs(filename);
    if(!ofs.is_open())return false;
    
    size_t layer_size = getNumberOfLayers();
    ofs << layer_size << std::endl;
    for(size_t i=0; i<layer_size; i++){
        ofs << getNumberOfUnits(i) << std::endl;
        for(auto&& unit: network[i]){
            ofs << unit.bias << std::endl;
            ofs << unit.weight.size() << std::endl;
            for(int j=0; j<unit.weight.size(); j++)ofs << ' ' << unit.weight[j];
            ofs << std::endl;
        }
    }
    
    return true;
}

bool FeedForward_Dy::importNetwork(std::string filename){
    
    std::ifstream ifs(filename);
    if(!ifs.is_open())return false;
    
    size_t layer_size,number_of_units,next_layer_size;
    ifs >> layer_size;
    network.resize(layer_size);
    
    try{
    
    for(size_t i=0; i<layer_size; i++){
        ifs >> number_of_units;
        network[i].resize(number_of_units);
        for(size_t j=0; j<number_of_units; j++){
            ifs >> network[i][j].bias;
            ifs >> next_layer_size;
            network[i][j].weight.resize(next_layer_size);
            for(size_t k=0; k<network[i][j].weight.size(); k++)ifs >> network[i][j].weight[k];
        }
    }
        
    }
    
    catch(std::exception& e){
        std::cout << e.what() << std::endl;
        return false;
    }
    
    return true;
}


std::vector<Unit_Dy>& FeedForward_Dy::layerForwardIterator(size_t layer_index,size_t unit_index){
	return network[layer_index+1];
}

std::vector<Unit_Dy>& FeedForward_Dy::layerBackwordIterator(size_t layer_index,size_t unit_index){
	return network[layer_index-1];
}

template<std::size_t W_SIZE>
class FeedForward_Amp {
public:
	std::vector< std::vector<Unit_Dy_Amp<W_SIZE>> > units;

	void setStruct(std::vector<unsigned int> number_of_units);

	bool exportNetwork(std::string filename);
	bool importNetwork(std::string filename);
};

template<std::size_t W_SIZE>
void FeedForward_Amp<W_SIZE>::setStruct(std::vector<unsigned int> number_of_units){

	units.resize(number_of_units.size());

	for (int i = 0; i < number_of_units.size(); i++) {

		units[i].resize(number_of_units[i]);

	}

}

template<std::size_t W_SIZE>
bool FeedForward_Amp<W_SIZE>::exportNetwork(std::string filename) {

	std::ofstream ofs(filename);
	if (!ofs.is_open())return false;

	size_t layer_size = units.size();
	ofs << layer_size << std::endl;
	for (size_t i = 0; i<layer_size; i++) {
		ofs << units[i].size() << std::endl;
		for (auto&& unit : units[i]) {
			ofs << unit.bias << std::endl;
			ofs << W_SIZE << std::endl;
			for (int j = 0; j< W_SIZE; j++)ofs << ' ' << unit.weight[j];
			ofs << std::endl;
		}
	}

	return true;
}

template<std::size_t W_SIZE>
bool FeedForward_Amp<W_SIZE>::importNetwork(std::string filename) {

	std::ifstream ifs(filename);
	if (!ifs.is_open())return false;

	size_t layer_size, number_of_units, next_layer_size;
	ifs >> layer_size;
	units.resize(layer_size);

	try {

		for (size_t i = 0; i<layer_size; i++) {
			ifs >> number_of_units;
			units[i].resize(number_of_units);
			for (size_t j = 0; j<number_of_units; j++) {
				ifs >> units[i][j].bias;
				ifs >> next_layer_size;
				//network[i][j].weight.resize(next_layer_size);
				for (size_t k = 0; k<W_SIZE; k++)ifs >> units[i][j].weight[k];
			}
		}

	}

	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return false;
	}

	return true;
}

template<std::size_t W_SIZE>
class FeedForward_Amp_View {
public:
	//Structure of network determined at the runtime
	typedef std::vector< std::vector<float> > structure;
	typedef FeedForward_Amp<W_SIZE> origin_data;
	typedef Unit_Dy_Amp<W_SIZE> Unit_t;
	typedef AMP tag;
	//C++ AMP-Restricted Function is not allow std::size_t(unsigned long)
	typedef unsigned int size_t;

	template<class ActivationObject>
	using Calc_Func = calcSurface<FeedForward_Amp_View, ActivationObject>;

	FeedForward_Amp_View(origin_data& network,Range);

	std::vector < concurrency::array_view<Unit_Dy_Amp<W_SIZE>> > network;
	void copy_to(origin_data&);
	void copy_from(origin_data&);

	size_t getNumberOfLayers(){ return network.size(); }
	size_t getNumberOfUnits(size_t layer_index) { return network[layer_index].get_extent()[0]; }

	Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& getLayer(size_t layer_index){ return network[layer_index]; };
	Unit_Dy_Amp<W_SIZE> getUnit(size_t layer_index, size_t unit_index){ return network[layer_index][unit_index]; };

	float* getWeight(size_t layer_index, size_t unit_index) { return network[layer_index][unit_index].weight; }
	float& getBias(size_t layer_index, size_t unit_index){ return network[layer_index][unit_index].bias }

	Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& layerForwardIterator(size_t layer_index, size_t unit_index); //forward iterator for propagation.
	Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& layerBackwordIterator(size_t layer_index, size_t unit_index);//backward iterator for propagation.

	bool exportNetwork(std::string filename);
	bool importNetwork(std::string filename);
private:
};

template<std::size_t W_SIZE>
FeedForward_Amp_View<W_SIZE>::FeedForward_Amp_View(origin_data& origin, Range range){
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<float> distribution(range.min_, range.max_);

	for (int i = 0; i < origin.units.size(); i++) {
		network.push_back(concurrency::array_view<Unit_t>(origin.units[i].size(), reinterpret_cast<Unit_t*>(&origin.units[i][0])));
	}

	for (size_t i = 0; i < getNumberOfLayers(); i++) {
		for (size_t j = 0; j < getNumberOfUnits(i); j++) {
			network[i][j].bias = 0;
			if (i == getNumberOfLayers() - 1) {
				for (int k = 0; k < Unit_t::W_SIZE; k++) {
					network[i][j].weight[k] = 0;
				}
			}
			else {
				for (int k = 0; k < Unit_t::W_SIZE; k++) {
					network[i][j].weight[k] = distribution(mt);
				}
			}
		}
	}
	
}

template<std::size_t W_SIZE>
void FeedForward_Amp_View<W_SIZE>::copy_to(origin_data& origin) {
	origin.units.resize( network.size() );
	for (int i = 0; i < network.size(); i++) {
		origin.units[i].resize(network[i].get_extent()[0]);
		auto& network_view = network[i];
		concurrency::array_view<Unit_Dy_Amp<W_SIZE>> units_view(origin.units[i].size(), reinterpret_cast<Unit_Dy_Amp<W_SIZE>*>(&origin.units[i][0]));
		parallel_for_each(network[i].get_extent(),[=](concurrency::index<1> idx)restrict(amp){
			units_view[idx] = network_view[idx];
		});
		units_view.synchronize();
	}
}

template<std::size_t W_SIZE>
void FeedForward_Amp_View<W_SIZE>::copy_from(origin_data& origin) {
	for (int i = 0; i < network.size(); i++) {
		auto& network_view = network[i];
		concurrency::array_view<Unit_Dy_Amp<W_SIZE>> units_view(origin.units[i].size(), reinterpret_cast<Unit_Dy_Amp<W_SIZE>*>(&origin.units[i][0]));
		parallel_for_each(network[i].get_extent(), [=](concurrency::index<1> idx)restrict(amp) {
			network_view[idx] = units_view[idx[0]];
		});
		network[i].synchronize();
	}
}

template<std::size_t W_SIZE>
Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& FeedForward_Amp_View<W_SIZE>::layerForwardIterator(size_t layer_index, size_t unit_index) {
	return network[layer_index + 1];
}

template<std::size_t W_SIZE>
Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& FeedForward_Amp_View<W_SIZE>::layerBackwordIterator(size_t layer_index, size_t unit_index) {
	return network[layer_index - 1];
}

template<std::size_t W_SIZE>
bool FeedForward_Amp_View<W_SIZE>::exportNetwork(std::string filename) {
	return true;
}

template<std::size_t W_SIZE>
bool FeedForward_Amp_View<W_SIZE>::importNetwork(std::string filename) {
	return true;
}

template<std::size_t Filter>
class FeedForward_Convolution{
public:
	//Structure of network determined at the runtime
	typedef std::vector< std::vector<float> > structure;
	typedef std::vector< std::vector<Size> > struct_t;
	typedef Map_Amp<Filter> Unit_t;
	typedef DYNAMIC tag;
	//C++ AMP-Restricted Function is not allow std::size_t(unsigned long)
	typedef unsigned int size_t;

	template<class ActivationObject>
	using Calc_Func = calcConvolution<FeedForward_Convolution, ActivationObject>;

	static constexpr size_t FilterSize = Filter;

	FeedForward_Convolution(const struct_t&,Range);

	std::vector< std::vector< Map_Amp<Filter> >  > network;
	std::vector< Size > sizes;

	void setSizeOfMap(size_t layer_index, size_t map_index, Size size) { 
		sizes[layer_index] = size;
		network[layer_index][map_index].setSizeOfMap(size); 
	};

	size_t getNumberOfLayers() { return network.size(); }
	size_t getNumberOfUnits(size_t layer_index) { return network[layer_index].size(); }
	Size getSizeOfMap(size_t layer_index) { return sizes[layer_index]; }

	std::vector<Map_Amp<Filter>>& getLayer(size_t layer_index) { return network[layer_index]; };
	Map_Amp<Filter> getUnit(size_t layer_index, size_t unit_index) { return network[layer_index][unit_index]; };

	std::vector<Map<Filter>>& getWeight(size_t layer_index, size_t unit_index) { return [layer_index][unit_index].weight; }
	float& getBias(size_t layer_index, size_t unit_index) { return network[layer_index][unit_index].bias }

	std::vector<Map_Amp<Filter>>& layerForwardIterator(size_t layer_index, size_t unit_index); //forward iterator for propagation.
	std::vector<Map_Amp<Filter>>& layerBackwordIterator(size_t layer_index, size_t unit_index);//backward iterator for propagation.

	bool exportNetwork(std::string filename);
	bool importNetwork(std::string filename);
private:
};

template<std::size_t Filter>
FeedForward_Convolution<Filter>::FeedForward_Convolution(const struct_t& size_of_maps,Range range) {
	network.resize( size_of_maps.size() );
	sizes.resize( size_of_maps.size() );
	for (size_t i = 0; i<size_of_maps.size(); i++) {
		network[i].resize(size_of_maps[i].size());
	}

	for (size_t i = 0; i<size_of_maps.size(); i++) {
		//todo autholize
		for (size_t j = 0; j<size_of_maps[i].size(); j++) {
			setSizeOfMap(i,j,size_of_maps[i][j]);
			network[i][j].weight.resize(i != size_of_maps.size() - 1 ? size_of_maps[i + 1].size() : 0);
		}
	}

	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_real_distribution<double> distribution(range.min_, range.max_);

	Map<Filter> zeros_map;
	for (auto&& cols : zeros_map) { for (auto&& unit : cols) unit = 0.0f; }

	for (size_t i = 0; i<getNumberOfLayers(); i++) {
		for (size_t j = 0; j<getNumberOfUnits(i); j++) {
			network[i][j].bias = 0;
			network[i][j].setSizeOfMap(size_of_maps[i][j]);

			if (i == getNumberOfLayers() - 1) {
				std::fill(network[i][j].weight.begin(), network[i][j].weight.end(), zeros_map);
			}
			else {
				Map<Filter> rnd_map;
				for (int idx = 0; idx < network[i][j].weight.size(); idx++) {
					for (auto&& cols : rnd_map) { for (auto&& unit : cols) unit = distribution(mt); }
					network[i][j].weight[idx] = rnd_map;
				}
			}
		}
	}
}

template<class NetworkStruct,class ActivationObject>
struct calcConvolution {
	void operator()(NetworkStruct& neural) {
		ActivationObject ao;
		for (int i = 1; i<neural.getNumberOfLayers(); i++) {
				std::vector< typename NetworkStruct::Unit_t >& layer = neural.network[i];
				std::vector< typename NetworkStruct::Unit_t >& back_layer = neural.network[i - 1];
				for (int j = 0; j < neural.getNumberOfUnits(i); j++) {
					//gpu acceleration
					const Size size = neural.getSizeOfMap(i);
					for (int k = 0; k < size.height; k++) {
						for (int l = 0; l < size.width; l++) {
							std::cout << "i: " << i << " j: " << j << " k: " << k << " l: " << l << std::endl;
							layer[j].setStatus(l,k,dot(back_layer, Point(l,k), j));
						}
					}
				}
		}
	}
	static float dot(const std::vector< typename NetworkStruct::Unit_t > & input_layer, Point map_index, size_t mapid){
		float sum = 0;
		const int f_size = NetworkStruct::FilterSize;
		float status;
		concurrency::array_view< float, 1 > s_view(1, &status);

		for (int i = 0; i < input_layer.size(); i++) {
			concurrency::array_view<const float, 2 > map( input_layer[i].size.height, input_layer[i].size.width, reinterpret_cast<const float*>(&input_layer[i].map[0]));
			concurrency::array_view<const float, 2> weight( f_size, f_size, reinterpret_cast<const float*>(&input_layer[i].weight[mapid][0][0]));

			/*parallel_for_each(weight.get_extent(), [=](concurrency::index<2> idx)restrict(amp) {
				s_view[0] += map[idx[1] + map_index.y][idx[0] + map_index.x] * weight[idx[1]][idx[0]];
			});*/
			for (int j = 0; j < f_size; j++) {
				for (int k = 0; k < f_size; k++) {
					std::cout << " j: " << j << " k: " << k << std::endl;
					s_view[0] += map[map_index.y + j][map_index.x + k] * weight[j][k];
				}
			}
			sum += ActivationObject::activate(s_view[0] + input_layer[i].bias);
		}
		return sum;
	}
};

template<class NetworkStruct, class ActivationObject>
struct _calcSurface<NetworkStruct,ActivationObject,DYNAMIC> {
	ActivationObject ao;
	void operator()(NetworkStruct& neural) {
		for (int i = 1; i<neural.getNumberOfLayers(); i++) {
			for (int j = 0; j<neural.getNumberOfUnits(i); j++) {
				neural.network[i][j].setStatus(ao.activate(sigma(neural.layerBackwordIterator(i, j), j) + neural.network[i][j].bias));
			}
		}
	}
	static double sigma(const std::vector<Unit_Dy>& input_layer, int unitid) {
		double sum = 0;
		for (auto& unit : input_layer) {
			sum += unit.getStatus() * unit.weight[unitid] + unit.bias;
		}
		return sum;
	}
};

template<class NetworkStruct, class ActivationObject>
struct _calcSurface<NetworkStruct, ActivationObject,AMP> {
	using Unit_t = typename NetworkStruct::Unit_t;
	void operator()(NetworkStruct& neural) {
		ActivationObject ao;
		for (int i = 1; i<neural.getNumberOfLayers(); i++) {
			concurrency::array_view<Unit_t>& layer = neural.network[i];
			concurrency::array_view<Unit_t>& back_layer = neural.network[i - 1];
			concurrency::extent<1> ex = layer.get_extent();
			//gpu acceleration
			parallel_for_each(ex, [=](concurrency::index<1> idx)restrict(amp) {
				layer[idx].setStatus(sigma(back_layer, idx[0]));
			});
		}
	}
	static float sigma(const concurrency::array_view<const typename NetworkStruct::Unit_t>& input_layer, int unitid)restrict(amp) {
		float sum = 0;
		for (int i = 0; i < input_layer.get_extent()[0]; i++) {
			sum += ActivationObject::activate_amp(input_layer[i].getStatus() + input_layer[i].bias) * input_layer[i].weight[unitid];
		}
		return sum;
	}
};


LIB_SCOPE_END()

namespace std{
    template< std::size_t I, std::size_t _First , std::size_t _Last , std::size_t... Args>
    constexpr auto&
    get( mtl::FeedForward<_First,_Last,Args...>& t ){
        return std::get<I>(t.network);
    }
    template<std::size_t _First, std::size_t _Last, std::size_t... Args>
    struct tuple_size<mtl::FeedForward<_First,_Last,Args...>>{
        static constexpr std::size_t value = mtl::FeedForward<Args...>::LAYER_SIZE;
    };
}


#endif /* defined(__MTL_Development__NNBase__) */
