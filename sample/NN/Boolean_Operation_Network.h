//
//  AND_NN.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/09/24.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_AND_NN_h
#define MTL_Development_AND_NN_h

#include"NNSolver.hpp"
#include<fstream>
#include<vector>
#include<utility>

/* Each module output the CSV file as result.*/

/* CSV format
 
x,y,z
INPUT1_1,INPUT1_2,OUTPUT1
INPUT2_1,INPUT2_2,OUTPUT2
...
INPUTn_1,INPUTn_2,OUTPUTn
 
 */

/* default input value step is 2 / 0.02 (100STEPS).*/

void and_nn(){
    mtl::NNSolver< mtl::FeedForward<2, 1>, mtl::tanh_af > solver;
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));
    
    solver.training<mtl::ErrorCorrection>(list, 100, 0.05);
    
    std::cout << "-------END--------" << std::endl;
    std::ofstream ofs("and_result.csv");
    ofs << "x," << "y," << "z" << std::endl;
    for(double x=-1.0; x<=1.0; x += 0.02){
        for(double y=-1.0; y<=1.0; y += 0.02){
            auto output = solver.solveAnswer( std::array<double,2>{x,y} );
            ofs << x << "," << y << "," << output[0].output(mtl::no_activation_af::activate) << std::endl;
        }
    }
    std::cout << std::endl;
}

void or_nn(){
    mtl::NNSolver< mtl::FeedForward<2, 1>,mtl::tanh_af > solver;
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));
    
    solver.training<mtl::ErrorCorrection>(list, 100, 0.05);
    
    std::cout << "-------END--------" << std::endl;
    std::ofstream ofs("or_result.csv");
    ofs << "x," << "y," << "z" << std::endl;
    for(double x=-1.0; x<=1.0; x += 0.02){
        for(double y=-1.0; y<=1.0; y += 0.02){
            auto output = solver.solveAnswer( std::array<double,2>{x,y} );
            ofs << x << "," << y << "," << output[0].output(mtl::no_activation_af::activate) << std::endl;
        }
    }
    std::cout << std::endl;
}

void xor_nn(){
    
    mtl::NNSolver< mtl::FeedForward<2,1,4>,mtl::tanh_af > solver;
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));

    
    solver.training<mtl::Backpropagation>(list, 1000, 0.01);
    
    std::cout << "-------END--------" << std::endl;
    std::ofstream ofs("xor_result.csv");
    ofs << "x," << "y," << "z" << std::endl;
    for(float x=-1.0; x<=1.0; x += 0.02){
        for(float y=-1.0; y<=1.0; y += 0.02){
            auto output = solver.solveAnswer( {x,y} );
            ofs << x << "," << y << "," << output[0].output(mtl::no_activation_af::activate) << std::endl;
        }
    }
    std::cout << std::endl;

    if(!solver.exportNetwork("xor_network_parameters.txt"))std::cout << "faild export" << std::endl;
}


#endif
