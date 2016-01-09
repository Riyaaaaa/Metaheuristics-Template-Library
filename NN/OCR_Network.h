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

std::vector< std::pair< std::vector<float> , std::vector<float> > > import_csv(std::string filename,const std::size_t InputVectorSize,const std::size_t OutputVectorSize){
    std::fstream filestream( filename );
    std::vector< std::pair< std::vector<float>, std::vector<float> > > list;
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
        std::vector<float> input(InputVectorSize);
        std::vector<float> output(OutputVectorSize);
        
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
    auto trainig_sample = import_csv("../ocr_test.csv",784,10);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp network;
	network.setStruct(network_struct);

    mtl::NNSolver< mtl::FeedForward_Amp_View, mtl::tanh_af_gpu_accel > solver(0.05,network);
    solver.training<mtl::Backpropagation_Gpu_Accel>(trainig_sample);
    
    std::cout << "-------END--------" << std::endl;
}

#endif
