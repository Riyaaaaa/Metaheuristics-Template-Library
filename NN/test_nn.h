//
//  test_nn.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/14.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_test_nn_h
#define MTL_Development_test_nn_h

#include"Boolean_Operation_Network.h"
#include"OCR_Network.h"

void unit_test(){
    mtl::NNSolver< mtl::FeedForward<1, 1>, mtl::sigmoid_af > solver(0.05);
    
    std::vector< std::pair< std::array<double,1>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,1>{1}, std::array<double,1>{0} ));
    
    solver.training<mtl::ErrorCorrection>(list);
    
    std::cout << "-------END--------" << std::endl;
}


void test_nn(){
    unit_test();
}

#endif
