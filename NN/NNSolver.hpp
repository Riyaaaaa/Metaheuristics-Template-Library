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
#include<utility>
#include<tuple>
#include<functional>
#include "../Multi_array.hpp"
#include "NNBase.h"

template<size_t begin, size_t end, bool terminate = begin + 1 == end>
struct ExecutePart;

template<size_t begin, size_t end>
struct ExecutePart<begin, end, true>
{
    template<typename Tuple, typename Function>
    static void Execute(Tuple && tuple, Function && function)
    {
        function(get<begin>(tuple),get<begin+1>(tuple));
    }
};

template<size_t begin, size_t end>
struct ExecutePart<begin, end, false>
{
    template<typename Tuple, typename Function>
    static void Execute(Tuple && tuple, Function && function)
    {
        // execute first half
        ExecutePart<begin, (begin + end) / 2>::Execute
        (forward<Tuple>(tuple), forward<Function>(function));
        
        // execute latter half
        ExecutePart<(begin + end) / 2, end>::Execute
        (forward<Tuple>(tuple), forward<Function>(function));
    }
};

// pass all element of tuple to function
template<typename Tuple, typename Function>
void ExecuteAll(Tuple && tuple, Function && function)
{
    using namespace std;
    
    static const size_t end =
    tuple_size<typename remove_reference<Tuple>::type>::value-1;
    
    ExecutePart<0, end>::Execute(
                                 forward<Tuple>(tuple),
                                 forward<Function>(function));
}

template <class Seq1, class Seq2>
struct concat;

template <class... Seq1, class... Seq2>
struct concat<std::tuple<Seq1...>, std::tuple<Seq2...>> {
    typedef std::tuple<Seq1..., Seq2...> type;
};

template<class T, class Tuple,std::size_t... Dims>
struct make_tuple_array{
    typedef Tuple type;
};

template<class T, class Tuple, std::size_t First, std::size_t... Dims>
struct make_tuple_array<T,Tuple,First,Dims...>{
    typedef typename make_tuple_array< T, typename concat < Tuple, std::tuple<std::array<T,First>>>::type  , Dims... >::type type;
};

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
class NNSolver {
public:
    typedef std::array<Unit , _First>    input_array;
    typedef std::array<Unit , _Last>     output_array;
    
    typename make_tuple_array<Unit, std::tuple<>, _First, Dims..., _Last>::type perceptron;

    output_array    solveAnswer(input_array);
    
private:
    
    struct calcSurface;
    
    template<std::size_t input_size, std::size_t output_size, std::size_t... Index>
    void sigma(std::array<Unit,input_size> input_surface);
    
    input_array     _sensory;
    output_array    _response;
    
    
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
};

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
typename NNSolver<_First , _Last , Dims...>::output_array NNSolver<_First , _Last , Dims...>::solveAnswer(input_array sensory){
    
    ExecuteAll(perceptron, calcSurface());
    
    _sensory = sensory;
    
    return output_array();
}

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
struct NNSolver<_First , _Last , Dims...>::calcSurface{
    template<std::size_t input_size, std::size_t output_size>
    void  operator()(std::array<Unit,input_size>& input_surface,
                     std::array<Unit,output_size>& output_surface){
        
        for(auto& n:input_surface){
            std::cout << n.weight << ' ';
        }
        std::cout << std::endl;
        
        for(auto& n:output_surface){
            std::cout << n.weight << ' ';
        }
        std::cout << std::endl;
        
                                                                 }
};

template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
template<std::size_t input_size, std::size_t output_size, std::size_t... Index>
void NNSolver<_First , _Last , Dims...>::sigma(std::array<Unit,input_size> input_surface){
    
}




#endif
