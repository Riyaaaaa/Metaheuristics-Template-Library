//
//  Multi_Array.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/13.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_Multi_Array_h
#define MTL_Development_Multi_Array_h

#include<array>

template<class T,std::size_t... Dims>
class Multi_Array{
public:
    typedef T  value_type;
    typedef T* iterator;
    typedef T  child;
    const std::size_t RANK = 0;
    T _value;
public:
    
    iterator begin(){
        return &_value;
    }
    iterator end(){
        return &(_value+1);
    }
    value_type& operator=(const value_type& rhs){
        _value = rhs;
        return _value;
    }
    operator value_type&(){
        return _value;
    }
};
template<class T,std::size_t First,std::size_t... Dims>
class Multi_Array<T,First,Dims...>{
public:
    typedef T  value_type;
    typedef T* iterator;
    typedef Multi_Array<T,Dims...> child;
    
public:
    static const std::size_t RANK = sizeof...(Dims);
    std::array<child , First> _array;
    
public:
    
    iterator begin(){
        return _array[0].begin();
    }
    iterator end(){
        return _array[First].begin();
    }
    child& operator[](const std::size_t& index){
        return _array[index];
    }
    
};

#endif
