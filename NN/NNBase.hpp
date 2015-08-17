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
#include "../Multi_array.hpp"

class Unit{
public:
    double weight=1;
    bool isBias=false;
    
    template<class F>
    double output(F&& f);
    
    template<std::size_t _iSize>
    double input(const std::array<Unit , _iSize>& surface);
    
    void setStatus(double _s){if(!isBias)_status = _s;}
private:
    double _status;
};

template<class F>
double Unit::output(F&& f){
    return f(weight*_status);
}

#endif /* defined(__MTL_Development__NNBase__) */
