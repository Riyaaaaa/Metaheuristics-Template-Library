//
//  OCR_Network.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/09/26.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_OCR_Network_h
#define MTL_Development_OCR_Network_h

#include"NNSolver.hpp"
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<utility>

template<std::size_t InputVectorSize,std::size_t OutputVectorSize>
std::vector< std::pair< std::array<double,InputVectorSize> , std::array<double,OutputVectorSize> > > import_csv(std::string filename){
    std::fstream filestream( filename );
    std::vector< std::pair< std::array<double,InputVectorSize>, std::array<double,OutputVectorSize> > > list;
    std::vector<std::vector<std::string>> table;
    const char delimiter = ',';
    std::string str;
    
    if (!filestream.is_open())
    {
        return list;
    }
    
    filestream >> str;
    
    while (!filestream.eof())
    {
        std::string buffer;
        filestream >> buffer;

        std::istringstream streambuffer(buffer);
        std::string token;
        std::array<double,InputVectorSize> input;
        std::array<double,OutputVectorSize> output;
        
        if(buffer.empty())break;
        
        std::fill(output.begin(),output.end(),0);
        
        getline(streambuffer, token, delimiter);
        output[std::stoi(token)]=1;
            
        for(int i=0; i<InputVectorSize; i++){
            getline(streambuffer, token, delimiter);
            input[i] = std::stoi(token);
        }
        list.push_back(std::make_pair(input,output));
    }
    
    for (int row = 0; row < list.size(); row++)
    {
        for (int column = 0; column < list[row].first.size(); column++)
        {
            if (column < list[row].first.size() - 1)
            {
            }
        }
        for (int column = 0; column < list[row].second.size(); column++)
        {
            if (column < list[row].second.size() - 1)
            {
            }
        }
    }
    
    return list;
}

void ocr_nn(){
    auto trainig_sample = import_csv<784,10>("../ocr_test.csv");
    mtl::NNSolver< mtl::FeedForward<784,10,784>, mtl::sigmoid_af > solver(0.05);
    
    solver.training<mtl::Backpropagation>(trainig_sample);
    
    std::cout << "-------END--------" << std::endl;
}

#endif
