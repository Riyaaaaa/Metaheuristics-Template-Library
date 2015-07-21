//
//  main.cpp
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/07.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <random>
//#include "SA/sample/test.h"
//#include "GA/sample/test.h"
#include "NN/NNSolver.hpp"
#include "NN/test_nn.h"

int main(int argc, const char * argv[]) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> distiribution(0,500);
    int city_size = 100;
    
    //tsp_individual a;
    
#ifdef OUTPUT
    
    ofstream ofs("./../../../../../MTL_Development/cities.txt");
    
    for(int i=0;i<city_size;i++){
        ofs << distiribution(mt) << ' ' << distiribution(mt) << std::endl;
    }
    
    /*
     for(int i=0; i<city_size; i++){
     city_list.push_back(cv::Point(250+200*cos(i*(2*3.14)/city_size),250+200*sin(i*(2*3.14)/city_size)));
     }
     */
    
#endif
    
    //test_ga();
    //test_sa();
    test_nn();
    return 0;
}
