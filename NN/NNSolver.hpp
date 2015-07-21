//
//  NNSolver.hpp
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/13.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_NNSolver_hpp
#define MTL_Development_NNSolver_hpp

#include<iostream>
#include<functional>
#include"Algorithm.hpp"
#include"NNBase.hpp"
#include"Utility.hpp"

namespace mtl{

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
class NNSolver {
public:
    explicit NNSolver(double t_rate):TRAINIG_RATE(t_rate){}
    
    typedef std::array<Unit , _First>    input_array;
    typedef std::array<Unit , _Last>     output_array;
    
    typedef typename mtl::make_tuple_array<Unit, std::tuple<>, _First, Dims..., _Last>::type network;
    
    network perceptron;
    
    typedef std::array<double,std::tuple_size<typename std::remove_reference< decltype(std::get<1>(perceptron)) >::type>::value > a;
    a _a;
    
    static constexpr int SURFACE_SIZE =  std::tuple_size<network>::value;
    const double TRAINIG_RATE;
    
    const output_array& solveAnswer(input_array);
    const output_array& training(input_array                                                input,
                                 std::array<double, std::tuple_size< output_array >::value>&& target);
    
private:
    
    struct calcSurface;
    
    input_array     _sensory;
    output_array    _response;
    
    /*
    template<typename Tuple, size_t... I>
    void NNapply_(Tuple& args, std::index_sequence<I...>)
    {
        calcSurface(std::get<I>(std::forward<Tuple>(args))...);
    }
    
    template<typename Tuple,
    typename Indices = std::make_index_sequence<std::tuple_size<Tuple>::value-1>>
    auto NNapply(Tuple& args)
    {
        NNapply_(args, Indices());
    }
     */
};

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
const typename NNSolver<_First , _Last , Dims...>::output_array& NNSolver<_First , _Last , Dims...>::solveAnswer(input_array sensory){
    
    std::get<0>(perceptron) = sensory;
    
    mtl::surfaceExecuteAll(perceptron, calcSurface());
    
    _sensory = sensory;
    
    return std::get< SURFACE_SIZE -1 >(perceptron);
}

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
const typename NNSolver<_First , _Last , Dims...>::output_array&
    NNSolver<_First , _Last , Dims...>::training(input_array                                                input,
                                                 std::array<double, std::tuple_size< output_array >::value>&& target){
    
    std::get<0>(perceptron) = input;
    mtl::surfaceExecuteAll(perceptron, calcSurface());
    
    Backpropagation _training;
        _training(std::move(perceptron),/*target*/ {1},TRAINIG_RATE);
        
    return std::get< SURFACE_SIZE-1 >(perceptron);
}

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
struct NNSolver<_First , _Last , Dims...>::calcSurface{
    template<std::size_t input_size, std::size_t output_size>
    void  operator()(std::array<Unit,input_size>& input_surface,
                     std::array<Unit,output_size>& output_surface){
        
        double sum = sigma(input_surface);
        
        for(auto& unit:output_surface){
            unit.status = sum;
        }
    }
    
    
    template<std::size_t input_size>
    double sigma(std::array<Unit,input_size>& input_surface){
        double sum=0;
        for(auto& unit: input_surface){
            sum += unit.output(sigmoid());
        }
        return sum;
    }
};

}

    



#endif
