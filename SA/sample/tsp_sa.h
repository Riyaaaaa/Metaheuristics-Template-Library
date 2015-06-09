//
//  tsp_sa.h
//  Procon26
//
//  Created by Riya.Liel on 2015/06/09.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __Procon26__tsp_sa__
#define __Procon26__tsp_sa__

#include "SABase.hpp"
#include "reference_list.h"

class tsp_annealing : public SA_Base<tsp_annealing, std::vector<cv::Point> ,std::vector<int> >{
public:
    tsp_annealing(std::vector<int> state):SA_Base<tsp_annealing, std::vector<cv::Point> ,std::vector<int> >(state){};

    tsp_annealing& turnState();
    int calcEvalution(std::vector<cv::Point>& city_list);
    
};

#endif /* defined(__Procon26__tsp_sa__) */
