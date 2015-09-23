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
#include<vector>
#include<utility>

void test_nn(){
    mtl::NNSolver< FeedForward<2, 1, 4> > a(0.05);
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));

    a.training<Backpropagation>(list);
    
    auto output = a.solveAnswer({0,1});
    
    std::cout << "-------result--------" << std::endl;
    for(auto& unit: output){
        std::cout << unit.output(threshold()) << ' ';
    }
    std::cout << std::endl;
}

#endif
