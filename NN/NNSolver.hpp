//
//  _NNSolver.hpp
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/13.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_NNSolver_hpp
#define MTL_Development_NNSolver_hpp

#include<iostream>
#include<functional>
#include<algorithm>
#include<vector>
#include<random>
#include<cmath>
#include"Algorithm.hpp"
#include"NNBase.hpp"
#include"Utility.hpp"
#include"../configuration.h"

LIB_SCOPE_BEGIN()

template<class Layer>
static double statusScanning(const Layer layer, const std::array<double, std::tuple_size<Layer>::value> target){
    double RMSerror=0.0;
    for(int i=0; i<std::tuple_size<Layer>::value ; i++){
        //RMSerror += 0.5 * std::pow(fabs(layer[i].getStatus() - target[i]),2);
        RMSerror += -target[i]*std::log(layer[i].getStatus())-(1-target[i])*std::log(1-layer[i].getStatus());
#ifdef DEBUG_MTL
        //std::cout << i+1 << "th units output " << layer[i].getStatus() << ", target value= " << target[i] << std::endl;
#endif
    }
    return RMSerror;
}

template<class Layer>
static double statusScanning(const Layer layer, const std::vector<float> target){
    double RMSerror=0.0;
    for(int i=0; i<target.size() ; i++){
        RMSerror += 0.5 * std::pow(fabs(layer[i].getStatus() - target[i]),2);
        //RMSerror += -target[i]*std::log(layer[i].getStatus())-(1-target[i])*std::log(1-layer[i].getStatus());
#ifdef DEBUG_MTL
        //std::cout << i+1 << "th units output " << layer[i].getStatus() << ", target value= " << target[i] << std::endl;
#endif
    }
    return RMSerror;
}

template<class Layer>
static void inputting(Layer& layer, const std::array<double, std::tuple_size<Layer>::value> input){
	for(int i=0; i<std::tuple_size<Layer>::value; i++){
		layer[i].setStatus(input[i]);
	}
}

template<class Layer>
static void inputting(Layer& layer, const std::vector<float> input){
    for(int i=0; i<input.size(); i++){
        layer[i].setStatus(input[i]);
    }
}

template<class NetworkStruct,class ActivationObject,class TAG>
class _NNSolver;

template<class NetworkStruct,class ActivationObject>
using NNSolver = _NNSolver<NetworkStruct,ActivationObject,typename NetworkStruct::tag>;

/********************************************************/
/*		 		static network solver					*/
/********************************************************/

template<class NetworkStruct,class ActivationObject>
class _NNSolver<NetworkStruct,ActivationObject,STATIC>{
	public:
		explicit _NNSolver(double t_rate);

		NetworkStruct neural;
#ifdef STATIC_NETWORK
		static constexpr std::size_t LAYER_SIZE = NetworkStruct::LAYER_SIZE;

		typedef typename NetworkStruct::template layer_type<LAYER_SIZE-1>   output_layer;
		typedef typename NetworkStruct::template layer_type<0>              input_layer;
#else
		static const std::size_t = NetworkStruct.getLayerSize();
		typedef typename NetworkStruct::output_layer output_layer;
		typedef typename NetworkStruct::input_layer input_layer;
#endif
		const double TRAINIG_RATE;

		auto solveAnswer(std::array<double, std::tuple_size<input_layer>::value>)
			->const output_layer;

		template<template<class,class>class _TRAINING_OBJECT>
			auto training(std::vector<
					std::pair<
					std::array<double, std::tuple_size<input_layer>::value>,
					std::array<double, std::tuple_size<output_layer>::value>
					>
					>& training_list)
			->const output_layer&;

		template<class _TRAINING_OBJECT>
			void regulateWeight(const std::array<double, std::tuple_size< input_layer >::value>& input,
					const std::array<double, std::tuple_size< output_layer >::value>& target,
					_TRAINING_OBJECT& _training_algorithm);

		bool exportNetwork(std::string filename);
	private:
		struct calcSurface;

		//input_layer     _sensory;
		//output_layer    _response;
};

template<class NetworkStruct,class ActivationObject>
_NNSolver<NetworkStruct,ActivationObject,STATIC>::_NNSolver(double t_rate):TRAINIG_RATE(t_rate){
	surfaceExecuteAll<0, LAYER_SIZE-1>(neural.network, [](auto& surface){
			std::random_device rnd;
			std::mt19937 mt(rnd());
			std::uniform_real_distribution<double> distribution(-1,1);
			for(int i=0; i<surface.size(); i++){
			surface[i].bias = 0;
			std::fill(surface[i].weight.begin(),surface[i].weight.end(),distribution(mt));
			}
			});
	auto& sensory = neural.template getLayer<LAYER_SIZE-1>();
	for(auto& unit: sensory){
		unit.bias = 0;
		std::fill(unit.weight.begin(),unit.weight.end(),0);
	}
}

	template<class NetworkStruct,class ActivationObject>
auto _NNSolver<NetworkStruct,ActivationObject,STATIC>::solveAnswer(const std::array<double, std::tuple_size<input_layer>::value> sensory)
	->const output_layer{

		inputting(std::get<0>(neural.network),sensory);

		mtl::static_for<1,LAYER_SIZE>(calcSurface(),neural);

		return std::get< LAYER_SIZE -1 >(neural.network);
	}


template<class NetworkStruct,class ActivationObject>
bool _NNSolver<NetworkStruct,ActivationObject,STATIC>::exportNetwork(std::string filename){

	std::ofstream ofs(filename);

	if(!ofs.is_open())return false;

	ofs << LAYER_SIZE << std::endl;

	forwardExecuteAll<0, LAYER_SIZE>(neural.network, [](auto& layer, std::ofstream& ofs){
			std::size_t layer_size = std::tuple_size<std::remove_reference_t<decltype(layer)>>::value;
			ofs << layer_size << std::endl;
			for(int i=0; i<layer_size; i++){
			ofs << layer[i].weight.size() << std::endl;
			ofs << layer[i].bias;
			for(int j=0; j<layer[i].weight.size(); j++)ofs << ' ' << layer[i].weight[j];
			ofs << std::endl;
			}
			},ofs);

	return true;
}

template<class NetworkStruct,class ActivationObject>
template<template<class,class>class _TRAINING_OBJECT>
auto _NNSolver<NetworkStruct,ActivationObject,STATIC>::training(std::vector<
		std::pair<
		std::array<double, std::tuple_size<input_layer>::value>,
		std::array<double, std::tuple_size<output_layer>::value>
		>
		>& training_list)
->const typename NetworkStruct::template layer_type<LAYER_SIZE-1>&{
	const std::size_t TRAINIG_LIMITS = 500;

	_TRAINING_OBJECT<NetworkStruct,ActivationObject> training_object;
	double RMSerror = 0.0, best = 1e6;
	NetworkStruct best_network;
	std::random_device rd;
	std::mt19937 mt(rd());

	for(int i=0; i<TRAINIG_LIMITS; i++){
#ifdef DEBUG_MTL
		//std::cout << "step " << i << std::endl;
#endif
		std::shuffle(training_list.begin(), training_list.end(),mt);
		for(auto& training_target: training_list){
			regulateWeight(training_target.first, training_target.second, training_object);
			RMSerror += statusScanning(solveAnswer(training_target.first),training_target.second);
		}
#ifdef DEBUG_MTL
		//std::cout << "RMSerror = " << RMSerror << std::endl;
		std::cout << i << ',' << RMSerror << std::endl;
#endif
		if(best > RMSerror){ best = RMSerror; best_network = neural;}
		RMSerror = 0.0;
	}
#ifdef DEBUG_MTL
	//std::cout << "best value = " << best << std::endl;
#endif
	neural = best_network;
	for(auto& training_target: training_list){
		regulateWeight(training_target.first, training_target.second, training_object);
		RMSerror += statusScanning(solveAnswer(training_target.first),training_target.second);
	}
	return std::get< LAYER_SIZE-1 >(neural);
}

template<class NetworkStruct,class ActivationObject>
template<class _TRAINING_OBJECT>
void _NNSolver<NetworkStruct,ActivationObject,STATIC>::regulateWeight(const std::array<double, std::tuple_size< input_layer >::value>& input,
		const std::array<double, std::tuple_size< output_layer >::value>& target,
		_TRAINING_OBJECT& _training_algorithm){
	inputting(std::get<0>(neural), input);
	static_for<1, LAYER_SIZE>(calcSurface(),neural);
	mtl::propagationTupleApply<LAYER_SIZE-1>(neural.network, _training_algorithm, target);

}

template<class NetworkStruct,class ActivationObject>
struct _NNSolver<NetworkStruct,ActivationObject,STATIC>::calcSurface{
	template<std::size_t index>
		void operator()(NetworkStruct& neural){
#ifdef STATIC_NETWORK
			static_for_nested<0,NetworkStruct::template getLayerSize<index>(),index>(unit_iterating(),neural);
#endif
		}

#ifdef STATIC_NETWORK
	struct unit_iterating{
		ActivationObject ao;
		template<std::size_t unit_index,std::size_t index,class T>
			void operator()(T& network){
				auto& unit = network.template getUnit<index,unit_index>();
				double sum = sigma(network.template layerBackwordIterator<index,unit_index>(),unit_index);
				unit.setStatus(ao.activate(sum+unit.bias));
			}
	};
#endif

	template<class Layer>
		static double sigma(const Layer& input_layer,int unitid){
			double sum=0;
			for(auto& unit: input_layer){
				sum += unit.getStatus() * unit.weight[unitid] + unit.bias;
			}
			return sum;
		}
};

/****************************************************/
/*				dynamic network solver				*/
/****************************************************/

template<class NetworkStruct,class ActivationObject>
class _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>{
public:
    explicit _NNSolver(double t_rate,std::vector<typename NetworkStruct::size_t> number_of_uints);
    explicit _NNSolver(double t_rate);
    
    NetworkStruct neural;
    typedef typename std::vector<Unit_Dy> output_layer;
    typedef typename std::vector<Unit_Dy> input_layer;
    
    const double TRAINIG_RATE;
    
    void setNumberOfUnitsOfLayers(std::vector<int> num_units);
    void setNumberOfUnitsByLayerIndex(int num_nutis);
    
    void setNetworkStruct(std::vector<typename NetworkStruct::size_t> number_of_uints);
    
    auto solveAnswer(std::vector<float>)
    ->const output_layer;
    
    template<template<class,class>class _TRAINING_OBJECT>
    auto training(std::vector<
                  std::pair<
                  std::vector<float>,
                  std::vector<float>
                  >
                  >& training_list)
    ->const output_layer&;
    
    template<class _TRAINING_OBJECT>
    void regulateWeight(const std::vector<float>& input,
                        const std::vector<float>& target,
                        _TRAINING_OBJECT& _training_algorithm);
    
    bool exportNetwork(std::string filename);
    bool importNetwork(std::string filename);
    
private:
    struct calcSurface;
};

template<class NetworkStruct,class ActivationObject>
_NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::_NNSolver(double t_rate,std::vector<typename NetworkStruct::size_t> number_of_uints):TRAINIG_RATE(t_rate){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> distribution(-1,1);
    
    setNetworkStruct(number_of_uints);
    
    for(typename NetworkStruct::size_t i=0; i<neural.getNumberOfLayers(); i++){
        for(typename NetworkStruct::size_t j=0; j<neural.getNumberOfUnits(i); j++){
            neural.network[i][j].bias = 0;
            if( i==neural.getNumberOfLayers()-1 ) {
                std::fill(neural.network[i][j].weight.begin(),neural.network[i][j].weight.end(),0);
            }
            else std::fill(neural.network[i][j].weight.begin(),neural.network[i][j].weight.end(),distribution(mt));
        }
    }
}
        
        template<class NetworkStruct,class ActivationObject>
        _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::_NNSolver(double t_rate):TRAINIG_RATE(t_rate){
        }

	template<class NetworkStruct,class ActivationObject>
auto _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::solveAnswer(const std::vector<float> sensory)
	->const output_layer{

		inputting(neural.network[0],sensory);

        calcSurface()(neural);

		return neural.network.back();
}

template<class NetworkStruct,class ActivationObject>
template<template<class,class>class _TRAINING_OBJECT>
auto _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::training(std::vector<
		std::pair<
		std::vector<float>,
		std::vector<float>
		>
		>& training_list)
->const output_layer&{
	const std::size_t TRAINIG_LIMITS = 500;

	_TRAINING_OBJECT<NetworkStruct,ActivationObject> training_object;
	double RMSerror = 0.0, best = 1e6;
	NetworkStruct best_network;
	std::random_device rd;
	std::mt19937 mt(rd());

	for(int i=0; i<TRAINIG_LIMITS; i++){
		std::shuffle(training_list.begin(), training_list.end(),mt);
		for(auto& training_target: training_list){
			regulateWeight(training_target.first, training_target.second, training_object);
			RMSerror += statusScanning(solveAnswer(training_target.first),training_target.second);
		}
#ifdef DEBUG_MTL
		std::cout << i << ',' << RMSerror << std::endl;
#endif
		if(best > RMSerror){ best = RMSerror; best_network = neural;}
		RMSerror = 0.0;
	}
#ifdef DEBUG_MTL
	//std::cout << "best value = " << best << std::endl;
#endif
	neural = best_network;
	for(auto& training_target: training_list){
		regulateWeight(training_target.first, training_target.second, training_object);
		RMSerror += statusScanning(solveAnswer(training_target.first),training_target.second);
	}
	return neural.network.back();
}

template<class NetworkStruct,class ActivationObject>
template<class _TRAINING_OBJECT>
void _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::regulateWeight(const std::vector<float>& input,
		const std::vector<float>& target,
		_TRAINING_OBJECT& _training_algorithm){
        inputting(neural.network.front(), input);
		calcSurface()(neural);
    
    auto delta = _training_algorithm(neural.network[neural.getNumberOfLayers()-1],target);
    for(int i=neural.getNumberOfLayers()-2; i>=0; i--){
        delta = _training_algorithm(neural.network[i],target,delta);
    }
}
        
        
template<class NetworkStruct,class ActivationObject>
        void _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::setNetworkStruct(std::vector<typename NetworkStruct::size_t> number_of_uints){
            
            neural.setNumberOfLayers( static_cast<typename NetworkStruct::size_t>(number_of_uints.size()) );
            for(typename NetworkStruct::size_t i=0; i<number_of_uints.size(); i++){
                neural.setNumberOfUnits( i , number_of_uints[i] );
            }
            
            for(typename NetworkStruct::size_t i=0; i<number_of_uints.size()-1; i++){
                //todo autholize
                for(typename NetworkStruct::size_t j=0; j<number_of_uints[i]; j++){
                    neural.network[i][j].weight.resize(number_of_uints[i+1]);
                }
            }
            
        }

template<class NetworkStruct,class ActivationObject>
struct _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::calcSurface{
    ActivationObject ao;
    void operator()(NetworkStruct& neural){
			for(int i=1; i<neural.getNumberOfLayers(); i++){
				for(int j=0; j<neural.getNumberOfUnits(i); j++){
					neural.network[i][j].setStatus(ao.activate(sigma(neural.layerBackwordIterator(i,j),j) + neural.network[i][j].bias));
				}
			}
		}
		static double sigma(const std::vector<Unit_Dy>& input_layer,int unitid){
			double sum=0;
			for(auto& unit: input_layer){
				sum += unit.getStatus() * unit.weight[unitid] + unit.bias;
			}
			return sum;
		}
};
        
        template<class NetworkStruct,class ActivationObject>
        bool _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::exportNetwork(std::string filename){
            return neural.exportNetwork(filename);
        }
        
        template<class NetworkStruct,class ActivationObject>
        bool _NNSolver<NetworkStruct,ActivationObject,DYNAMIC>::importNetwork(std::string filename){
            return neural.importNetwork(filename);
        }

LIB_SCOPE_END()

#endif
