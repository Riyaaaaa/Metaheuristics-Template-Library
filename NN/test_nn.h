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
    NNSolver<3, 1, 2> a;
    a.solveAnswer({1,1,1});
}

#endif
