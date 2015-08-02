//
//  test_nn.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/14.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_test_nn_h
#define MTL_Development_test_nn_h

#include"NNSolver.hpp"

void test_nn(){
    mtl::NNSolver<2, 1, 2> a(0.1);
    
    a.training({1,1}, {1});
    a.training({0,1}, {0});
    a.training({1,0}, {0});
    a.training({0,0}, {0});
    
    auto output = a.solveAnswer({1,1});
    
    for(auto& unit: output){
        std::cout << unit.output(sigmoid()) << ' ';
    }
    std::cout << std::endl;
}

#endif
