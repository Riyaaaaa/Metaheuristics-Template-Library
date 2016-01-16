//
//  NNBase.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/13.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __MTL_Development__NNBase__
#define __MTL_Development__NNBase__

#include <array>
#include <amp.h>
#include <utility>
#include <fstream>
#include "Utility.hpp"
#include "../configuration.h"

LIB_SCOPE_BEGIN()

template<std::size_t WEIGHT_SIZE>
class FeedForward_Amp;

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

	void    setStatus(float _s) restrict(cpu,amp){ _status = _s; }
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

/* Feed forward perceptron class */
/* This is the Perceptron model that does not have the recursive structure. */
/* This is implemented by std::tuple. Therefore, this require tuple mtl-utility(See Utility.hpp).  */

struct STATIC{};
struct DYNAMIC{};
struct AMP {};

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
	typedef DYNAMIC tag;
	//C++ AMP-Restricted Function is not allow std::size_t(unsigned long)
	typedef unsigned int size_t;
    
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

	std::vector < concurrency::array_view<Unit_Dy_Amp<W_SIZE>> > network;
	void copy_to(origin_data&);
	void copy_from(origin_data&);

	size_t getNumberOfLayers(){ return network.size(); }
	size_t getNumberOfUnits(size_t layer_index) { return network[layer_index].get_extent()[0]; }

	Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& getLayer(size_t layer_index){ return network[layer_index]; };
	Unit_Dy_Amp<W_SIZE> getUnit(size_t layer_index, size_t unit_index){ return network[layer_index][unit_index]; };

	Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& layerForwardIterator(size_t layer_index, size_t unit_index); //forward iterator for propagation.
	Concurrency::array_view<Unit_Dy_Amp<W_SIZE>>& layerBackwordIterator(size_t layer_index, size_t unit_index);//backward iterator for propagation.

	bool exportNetwork(std::string filename);
	bool importNetwork(std::string filename);
private:
};

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
