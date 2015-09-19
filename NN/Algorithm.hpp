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

struct sigmoid{
    double operator()(double input,double a=1)  { return 1 / (1. + exp(-a*input));}
};

struct threshold{
    double operator()(double input,double T=0.5){ return input > T ? 1 : 0; }
};

template<class Tuple>
struct ErrorCorrection{
    typedef std::remove_reference_t<Tuple> Tuple_t;
    typedef std::array<double,std::tuple_size<typename std::tuple_element<std::tuple_size<Tuple_t>::value-1,Tuple_t>::type >::value> output_layer_t;
    
    const double _trate = 0.15;
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit<Size2>,Size1>& layer,output_layer_t&& target){
        double out,act_dot_delta=0;
        std::array<double,std::tuple_size<output_layer_t>::value> delta;
        
        for(std::size_t i=0; i<target.size() ; i++){
            out = layer[i].output(sigmoid());
            delta[i] = out * (1 - out) * (target[i] - out);

            act_dot_delta += delta[i] * out;
        }
        
        for(int i=0; i<layer.size(); i++){
            for(int j=0; j<Size2; j++){
                layer[i].bias += delta[j];
                layer[i].weight[j] += act_dot_delta;
            }
        }
        
        return delta;
    }
    
    template<std::size_t Size1,std::size_t Size2>
    void operator()(std::array<Unit<Size2>,Size1>& input_layer,output_layer_t&& target, std::array<double,Size2>&& delta){
        
        for(auto& unit: input_layer){
            for(int i=0; i<Size2; i++){
                unit.weight[i] += _trate * delta[i] * unit.output(sigmoid());
                unit.bias += _trate * delta[i];
            }
        }
    }
};

template<class Tuple, bool isSensory = (std::tuple_size<Tuple>::value > 2) >
struct _Backpropagation;
/*  Back propagation requires three or more layers */

template<class Tuple>
using Backpropagation = _Backpropagation<Tuple>;

template<class Tuple>
struct _Backpropagation<Tuple,true>{
    typedef std::remove_reference_t<Tuple> Tuple_t;
    typedef std::array<double,std::tuple_size<typename std::tuple_element<std::tuple_size<Tuple_t>::value-1,Tuple_t>::type >::value> output_layer_t;
    
    const double _trate = 0.15;
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit<Size2>,Size1>& layer,output_layer_t&& target){
        double out,act_dot_delta=0;
        std::array<double,std::tuple_size<output_layer_t>::value> delta;
        
        for(std::size_t i=0; i<target.size() ; i++){
            out = layer[i].output(sigmoid());
            delta[i] = out * (1 - out) * (target[i] - out);
            
            act_dot_delta += delta[i] * out;
        }
        
        for(int i=0; i<layer.size(); i++){
            for(int j=0; j<Size2; j++){
                layer[i].bias += delta[j];
                layer[i].weight[j] += act_dot_delta;
            }
        }
        
        return delta;
    }
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit<Size2>,Size1>& input_layer,output_layer_t&& target, std::array<double,Size2>&& delta){

        double out, propagation=0;
        
        std::array<double,Size1> new_delta;
        
        for(auto& unit: input_layer){
            for(int i=0; i<Size2; i++){
                unit.weight[i] += _trate * delta[i] * unit.output(sigmoid());
                unit.bias += _trate * delta[i];
            }
        }
        
        for(int j=0; j<Size1; j++){
            for(int i=0; i<Size2; i++){
                propagation += input_layer[j].weight[i] * delta[i];
            }
            out = input_layer[j].output(sigmoid());
            new_delta[j] = out * (1-out) * propagation;
        }
        
        return new_delta;
    }
};

/* Network does not have three or more layers */
template<class Tuple>
struct _Backpropagation<Tuple,false>;

#endif
