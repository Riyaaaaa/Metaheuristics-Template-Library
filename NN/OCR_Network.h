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
        // ファイルが開けなかった場合は終了する
        return list;
    }
    
    filestream >> str;
    
    while (!filestream.eof())
    {
        // １行読み込む
        std::string buffer;
        filestream >> buffer;
        
        // ファイルから読み込んだ１行の文字列を区切り文字で分けてリストに追加する
        std::istringstream streambuffer(buffer); // 文字列ストリーム
        std::string token;                       // １セル分の文字列
        std::array<double,InputVectorSize> input;
        std::array<double,OutputVectorSize> output;
        
        if(buffer.empty())break;
        
        for(int i=0; i<OutputVectorSize; i++){
            getline(streambuffer, token, delimiter);
            output[i] = std::stoi(token);
        }
        for(int i=0; i<InputVectorSize; i++){
            getline(streambuffer, token, delimiter);
            input[i] = std::stoi(token);
        }
        // １行分の文字列を出力引数のリストに追加する
        list.push_back(std::make_pair(input,output));
    }
    
    for (int row = 0; row < list.size(); row++)
    {
        // １セルずつ読み込んでコンソールに出力する
        for (int column = 0; column < list[row].first.size(); column++)
        {
            std::cout << list[row].first[column];
            // 末尾の列でない場合はカンマを出力する
            if (column < list[row].first.size() - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
        for (int column = 0; column < list[row].second.size(); column++)
        {
            std::cout << list[row].second[column];
            // 末尾の列でない場合はカンマを出力する
            if (column < list[row].second.size() - 1)
            {
                std::cout << ",";
            }
        }
        std::cout << std::endl;
    }
    
    return list;
}

void ocr_nn(){
    auto trainig_sample = import_csv<784,1>("ocr_test.csv");
}

#endif
