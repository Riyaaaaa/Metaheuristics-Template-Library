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
#include"NNBase.hpp"

struct sigmoid{
    double operator()(double input,double a=1){return 1 / 1 + exp(-a*input);}
};

struct Backpropagation{
    /*
    template<std::size_t _Size1, std::size_t _Size2>
    void operator()(std::array<Unit, _Size1>&& surface1, std::array<Unit, _Size2> surface2,double delta,double training_rate){
        for(auto& unit: surface1){
            unit.weight += training_rate * delta * unit.output(sigmoid());
        }
    }
     */
    
    double _trate;
    
    template<class Tuple>
    void operator()(Tuple&& perceptron,
                    std::array<double,
                    std::tuple_size< typename std::remove_reference <decltype(std::get<std::tuple_size<Tuple>::value-1>(perceptron))>::type >::value>&& target,
                    double training_rate){
        
        static constexpr std::size_t SURFACE_SIZE = std::tuple_size<Tuple>::value;
        typedef std::remove_reference<decltype(std::get< SURFACE_SIZE-1 >(perceptron))> output_array;
        typedef std::remove_reference<decltype(std::get< SURFACE_SIZE-2 >(perceptron))> input_array;
        auto& response = std::get< SURFACE_SIZE-1 >(perceptron);
        auto& sensory = std::get< SURFACE_SIZE-2 >(perceptron);
        
        typename std::remove_reference<decltype(target)>::type delta;
        
        double out_delta,out;
        _trate = training_rate;
        
        for(std::size_t i=0; i<target.size() ; i++){
            out = response[i].output(sigmoid());
            out_delta = out * (1 - out) * (target[i] - out);
        
            std::size_t input_array_size = sensory.size();
            for(int j=0; j<input_array_size; j++){
                delta[j] = _trate * out_delta * sensory[j].output(sigmoid());
                sensory[j].weight += delta[j];
            }
        }
        
        unfoldTuple<Tuple,SURFACE_SIZE-3>(std::forward<Tuple>(perceptron), std::forward<decltype(delta)>(delta));
        
    }
    
    template<class Tuple,std::size_t index>
    void unfoldTuple(Tuple&& perceptron,
                     std::array<double,
                     std::tuple_size< typename std::remove_reference <decltype(std::get<std::tuple_size<Tuple>::value-1>(perceptron))>::type >::value>&& delta){
        
        auto& out_surface = std::get<index+1>(perceptron);
        auto& input_surface = std::get<index>(perceptron);
        std::size_t out_array_size = std::tuple_size<typename std::remove_reference<decltype(out_surface)>::type>::value;
        std::size_t input_array_size = std::tuple_size<typename std::remove_reference<decltype(input_surface)>::type>::value;
        decltype(std::get< index+1 >(perceptron)) new_delta;

        double out, propagation=0;
        
        for(int j=0; j<input_array_size; j++){
            for(int i=0; i<out_array_size; i++){
                propagation += input_surface[j].weight * delta[i];
            }
            out = input_surface[j].output(sigmoid());
            new_delta[j] = out * (1-out) * propagation;
        }
        
        unfoldTuple<Tuple,index-1>(perceptron, new_delta);
    }
    
    template<class Tuple>
    void unfoldTuple<0>(Tuple&& perceptron,
                              std::array<double,
                              std::tuple_size< typename std::remove_reference <decltype(std::get<std::tuple_size<Tuple>::value-1>(perceptron))>::type >::value>&& delta);
    
};
#endif
