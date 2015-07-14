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

template<class T, class Tuple,std::size_t... Dims>
struct make_tuple_array{
    typedef Tuple type;
};

template<class T, class Tuple, std::size_t First, std::size_t... Dims>
struct make_tuple_array<T,Tuple,First,Dims...>{
    typedef typename make_tuple_array< T, typename mtl::concat < Tuple, std::tuple<std::array<T,First>>>::type  , Dims... >::type type;
};

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
class NNSolver {
public:
    typedef std::array<Unit , _First>    input_array;
    typedef std::array<Unit , _Last>     output_array;
    
    typedef typename make_tuple_array<Unit, std::tuple<>, _First, Dims..., _Last>::type network;
    network perceptron;

    output_array    solveAnswer(input_array);
    
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
typename NNSolver<_First , _Last , Dims...>::output_array NNSolver<_First , _Last , Dims...>::solveAnswer(input_array sensory){
    
    std::get<0>(perceptron) = sensory;
    
    mtl::ExecuteAll(perceptron, calcSurface());
    
    _sensory = sensory;
    
    return std::get< std::tuple_size<network>::value -1 >(perceptron);
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

    



#endif
