//
//  tsp_sa.cpp
//  Procon26
//
//  Created by Riya.Liel on 2015/06/09.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#include "tsp_sa.h"

tsp_annealing& tsp_annealing::turnState(){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_real_distribution<double> distribution(0,_state.size()-1);
    
    int cross_point1 = distribution(mt);
    int cross_point2 = mt() % (_state.size()-cross_point1);
    
    for(int i=cross_point1; i<=cross_point1 + (cross_point2-cross_point1)/2; i++){
        std::swap(_state[i],_state[cross_point1+cross_point2-i]);
    }
    
    return *this;
}

int tsp_annealing::calcEvalution(std::vector<cv::Point> &city_list){
    int evalution=0,std_eval=0;
    
    for(int i=0;i<city_list.size();i++){
        std_eval += cv::norm(cv::Point(0,0)-city_list[i]);
    }
    
    for(int i=0;i<city_list.size()-1;i++){
        evalution+= cv::norm(city_list[_state[i]]-city_list[_state[i+1]]);
    }
    evalution+= cv::norm(city_list[_state.back()]-city_list[_state.front()]);
    
    _evalution = std_eval - evalution;
    
    return _evalution > 0 ? _evalution : 0;
}