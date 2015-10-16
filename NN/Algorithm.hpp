//
//  Algorithm.hpp
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/14.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_Algorithm_hpp
#define MTL_Development_Algorithm_hpp

#include<cmath>
#include"Utility.hpp"
#include"NNBase.hpp"
#include"../configuration.h"

LIB_SCOPE_BEGIN()

struct threshold_af : ActivationFunc<threshold_af>{
    static double activate(double input,double T=0){ return input > T ? 1 : -1; }
    static double activateDerivative(double input); //no definition
};

struct rectified_linear_units_af : ActivationFunc<rectified_linear_units_af>{
    static double activate(double input){ return input >= 0 ? input : 0; }
    static double activateDerivative(double input); //no definition
};

struct no_activation_af : ActivationFunc<no_activation_af>{
    static double activate(double input){ return input; }
    static double activateDerivative(double input); //no definition
};

struct sigmoid_af : ActivationFunc<sigmoid_af>{
    static double activate(double input){ return 1 / (1. + exp(-input));}
    static double activateDerivative(double input){ return input * (1 - input);}
};

struct tanh_af : ActivationFunc<mtl::tanh_af>{
    static double activate(double input){ return tanh(input);}
    static double activateDerivative(double input){ return 1 - input*input;}
};

template<class Tuple,class ActivationObject>
struct ErrorCorrection{
    typedef std::remove_reference_t<Tuple> Tuple_t;
    typedef std::array<double,std::tuple_size<typename std::tuple_element<std::tuple_size<Tuple_t>::value-1,Tuple_t>::type >::value> output_layer_t;
    typedef ActivationObject actiavation_type;
    
    const double _trate = 0.15;
    actiavation_type ao;
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit<Size2>,Size1>& layer,output_layer_t& target){
        double out;
        std::array<double,std::tuple_size<output_layer_t>::value> delta;
        
        for(std::size_t i=0; i<target.size() ; i++){
            out = layer[i].getStatus();
            //delta[i] = ao.activateDerivative(out) * (target[i] - out);
            delta[i] = (out - target[i]);
            layer[i].bias -= _trate * delta[i];
        }
        
        return delta;
    }
    
    template<std::size_t Size1,std::size_t Size2>
    void operator()(std::array<Unit<Size2>,Size1>& input_layer,output_layer_t& target, std::array<double,Size2>&& delta){
        for(auto& unit: input_layer){
            for(int i=0; i<Size2; i++){
                unit.weight[i] -= _trate * delta[i] * unit.getStatus();
            }
        }
    }
};

template<class Tuple,class ActivationObject,bool isSensory = (std::tuple_size<Tuple>::value > 2) >
struct _Backpropagation;
/*  Back propagation requires three or more layers */

template<class Tuple,class ActivationObject>
using Backpropagation = _Backpropagation<Tuple,ActivationObject>;

template<class Tuple,class ActivationObject>
struct _Backpropagation<Tuple,ActivationObject,true>{
    typedef std::remove_reference_t<Tuple> Tuple_t;
    typedef std::array<double,std::tuple_size<typename std::tuple_element<std::tuple_size<Tuple_t>::value-1,Tuple_t>::type >::value> output_layer_t;
    
    const double _trate = 0.15;
    ActivationObject ao;
    
    _Backpropagation(){
        std::cout << "test" << std::endl;
    }
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit<Size2>,Size1>& layer,output_layer_t& target){
        double out;
        std::array<double,std::tuple_size<output_layer_t>::value> delta;
        
        for(std::size_t i=0; i<target.size() ; i++){
            out = layer[i].getStatus();
            delta[i] = ao.activateDerivative(out) * (target[i] - out);
            layer[i].bias += _trate * delta[i];
        }
        
        return delta;
    }
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit<Size2>,Size1>& input_layer,output_layer_t& target, std::array<double,Size2>&& delta){

        double out, propagation=0;
        
        std::array<double,Size1> new_delta;
        
        for(auto& unit: input_layer){
            for(int i=0; i<Size2; i++){
                unit.weight[i] += _trate * delta[i] * unit.getStatus();
            }
        }
        
        for(int j=0; j<Size1; j++){
            for(int i=0; i<Size2; i++){
                propagation += input_layer[j].weight[i] * delta[i];
            }
            out = input_layer[j].getStatus();
            new_delta[j] = ao.activateDerivative(out) * propagation;
            input_layer[j].bias += _trate * new_delta[j];
        }
        
        return new_delta;
    }
};

/* Network does not have three or more layers */
template<class Tuple,class ActivationObject>
struct _Backpropagation<Tuple,ActivationObject,false>;

LIB_SCOPE_END()
    
#endif

