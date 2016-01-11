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
        
        std::fill(output.begin(),output.end(),0.f);
        
        getline(streambuffer, token, delimiter);
        output[std::stoi(token)]=1;
        
        for(int i=0; i<InputVectorSize; i++){
            getline(streambuffer, token, delimiter);
            input[i] = std::stoi(token) / 128.f - 1;
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

void ocr_nn(std::string filename){
    auto trainig_sample = import_csv(filename,784,10);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp network;
	network.setStruct(network_struct);
    mtl::NNSolver< mtl::FeedForward_Amp_View, mtl::tanh_af_gpu_accel > solver(0.05,network);
	network.exportNetwork("ocr_network.txt");
	//solver.setNetworkStruct(network_struct);
    solver.training<mtl::Backpropagation_Gpu_Accel>(trainig_sample);
	solver.exportNetwork("ocr_network.txt");
	//network.exportNetwork("ocr_network.txt");
    
    std::cout << "-------END--------" << std::endl;
}

void ocr_nn(std::string csv_filename,std::string network_filename) {
	auto trainig_sample = import_csv(csv_filename, 784, 10);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp network;
	network.setStruct(network_struct);
	mtl::NNSolver< mtl::FeedForward_Amp_View, mtl::tanh_af_gpu_accel > solver(0.05, network);
	network.importNetwork(network_filename);
	//solver.setNetworkStruct(network_struct);
	solver.training<mtl::Backpropagation_Gpu_Accel>(trainig_sample);
	solver.exportNetwork("ocr_network.txt");
	//network.exportNetwork("ocr_network.txt");

	std::cout << "-------END--------" << std::endl;
}

void ocr_sample_trimmer(int scale) {
	auto training_sample = import_csv("../../NN/training_sample/ocr_test.csv",784,10);
	std::ofstream ofs("ocr_test_scale_" + std::to_string(scale) + ".csv");

	ofs << "label";
	for (int i = 0; i < training_sample[0].first.size(); i++) {
		ofs << "pixel" + std::to_string(i);
		if (i != training_sample[0].first.size() - 1)ofs << ",";
	}

	ofs << std::endl;

	for (int i = 0; i < training_sample.size(); i+=scale) {
		for (int j = 0; j < training_sample[i].second.size(); j++) {
			float label = training_sample[i].second[j];
			if (label == 1) { ofs << j; break;  }
		}
		for (int j = 0; j < training_sample[i].first.size(); j++) {
			ofs << "," << training_sample[i].first[j];
		}
		ofs << std::endl;
	}
}

#endif
