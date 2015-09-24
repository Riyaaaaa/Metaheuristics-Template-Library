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

void and_nn(){
    mtl::NNSolver< FeedForward<2, 1> > solver(0.05);
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));
    
    solver.training<ErrorCorrection>(list);
    
    std::cout << "-------result--------" << std::endl;
    std::ofstream ofs("and_result.csv");
    ofs << "x," << "y," << "z" << std::endl;
    for(double x=-1.0; x<=1.0; x += 0.02){
        for(double y=-1.0; y<=1.0; y += 0.02){
            auto output = solver.solveAnswer( std::array<double,2>{x,y} );
            ofs << x << "," << y << "," << output[0].output(no_activation()) << std::endl;
        }
    }
    std::cout << std::endl;
}

void or_nn(){
    mtl::NNSolver< FeedForward<2, 1> > solver(0.05);
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));
    
    solver.training<ErrorCorrection>(list);
    
    std::cout << "-------result--------" << std::endl;
    std::ofstream ofs("or_result.csv");
    ofs << "x," << "y," << "z" << std::endl;
    for(double x=-1.0; x<=1.0; x += 0.02){
        for(double y=-1.0; y<=1.0; y += 0.02){
            auto output = solver.solveAnswer( std::array<double,2>{x,y} );
            ofs << x << "," << y << "," << output[0].output(no_activation()) << std::endl;
        }
    }
    std::cout << std::endl;
}

void xor_nn(){
    mtl::NNSolver< FeedForward<2, 1, 4> > solver(0.05);
    
    std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
    list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{-1} ));
    list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{1} ));
    list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));
    
    solver.training<Backpropagation>(list);
    
    std::cout << "-------result--------" << std::endl;
    std::ofstream ofs("result.csv");
    ofs << "x," << "y," << "z" << std::endl;
    for(double x=-1.0; x<=1.0; x += 0.02){
        for(double y=-1.0; y<=1.0; y += 0.02){
            auto output = solver.solveAnswer( std::array<double,2>{x,y} );
            ofs << x << "," << y << "," << output[0].output(no_activation()) << std::endl;
        }
    }
    std::cout << std::endl;
}


#endif
