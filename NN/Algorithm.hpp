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
    double operator()(double input,double a=1){return 1 / 1 + exp(-a*input);}
};

template<class Tuple, bool isSensory = (std::tuple_size<Tuple>::value > 2) >
struct Backpropagation;

template<class Tuple>
struct Backpropagation<Tuple,true>{
    /*
    template<std::size_t _Size1, std::size_t _Size2>
    void operator()(std::array<Unit, _Size1>&& surface1, std::array<Unit, _Size2> surface2,double delta,double training_rate){
        for(auto& unit: surface1){
            unit.weight += training_rate * delta * unit.output(sigmoid());
        }
    }
     */
    
    double _trate;
    
    void PropagationApply(Tuple&& perceptron,
                    std::array<double,
                    std::tuple_size< std::remove_reference_t <decltype(std::get<std::tuple_size<Tuple>::value-1>(perceptron))> >::value>&& target,
                    double training_rate){
        
        static constexpr std::size_t SURFACE_SIZE = std::tuple_size<Tuple>::value;
        typedef std::remove_reference_t<decltype(std::get< SURFACE_SIZE-1 >(perceptron))> output_array;
        typedef std::remove_reference_t<decltype(std::get< SURFACE_SIZE-2 >(perceptron))> input_array;
        auto& response = std::get< SURFACE_SIZE-1 >(perceptron);
        auto& sensory = std::get< SURFACE_SIZE-2 >(perceptron);

        std::array<double, std::tuple_size< std::remove_reference_t< decltype(sensory) > > ::value > delta;
        
        double out_delta,out;
        _trate = training_rate;
        
        for(std::size_t i=0; i<target.size() ; i++){
            out = response[i].output(sigmoid());
            out_delta = out * (1 - out) * (target[i] - out);
        
            for(int j=0; j<delta.size(); j++){
                delta[j] = _trate * out_delta * sensory[j].output(sigmoid());
                sensory[j].weight += delta[j];
            }
        }
        
        mtl::propagationTuple<SURFACE_SIZE-3>::Execute(std::forward<Tuple>(perceptron), *this, std::forward<decltype(delta)>(delta));
        
    }
    
    template<std::size_t Size1,std::size_t Size2>
    std::array<double,Size1> operator()(std::array<Unit,Size1>& input_surface, std::array<double,Size2>& delta){

        double out, propagation=0;
        
        std::array<double,Size1> new_delta;
        
        for(int j=0; j<Size1; j++){
            for(int i=0; i<Size2; i++){
                propagation += input_surface[j].weight * delta[i];
            }
            out = input_surface[j].output(sigmoid());
            new_delta[j] = out * (1-out) * propagation;
        }
        
        return new_delta;
    }
};

template<class Tuple>
struct Backpropagation<Tuple,false>;

#endif
