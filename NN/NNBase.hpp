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
#include <utility>
#include "Utility.hpp"
#include "../configuration.h"

LIB_SCOPE_BEGIN()

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
    
    template<std::size_t _iSize>
    double input(const std::array<Unit , _iSize>& surface);
    
    void    setStatus(double _s){_status = _s;}
    double  getStatus(){return _status;}
private:
    float _status;
};

template<std::size_t _NEXT_LAYER_SIZE>
template<class F>
double Unit<_NEXT_LAYER_SIZE>::output(F&& f){
    return f(_status+bias);
}

/* Feed forward perceptron class */
/* This is the Perceptron model that does not have the recursive structure. */
/* This is implemented by std::tuple. Therefore, this require tuple mtl-utility(See Utility.hpp).  */

template<std::size_t _First , std::size_t _Last , std::size_t... Args>
class FeedForward{
public:
    typedef typename mtl::make_tuple_array_3dims<Unit, std::tuple<>, _First, Args..., _Last>::type structure; //network structure
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


LIB_SCOPE_END()

namespace std{
    template< std::size_t I, std::size_t _First , std::size_t _Last , std::size_t... Args>
    constexpr auto&
    get( mtl::FeedForward<_First,_Last,Args...>& t ){
        return std::get<I>(t.network);
    }
    template<std::size_t... Args>
    struct tuple_size<mtl::FeedForward<Args...>>{
        static constexpr std::size_t value = mtl::FeedForward<Args...>::LAYER_SIZE;
    };
}


#endif /* defined(__MTL_Development__NNBase__) */
