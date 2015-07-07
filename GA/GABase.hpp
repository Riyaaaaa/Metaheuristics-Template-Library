//
//  Structure.h
//  opencv_test
//
//  Created by Riya.Liel on 2015/04/24.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __GA_GA_Base__
#define __GA_GA_Base__

#include <vector>
#include <array>

class GA_Base{ //Interface Template class for GA
public:
    
    /* Functions that you must define */
    
    virtual GA_Base* mutation()=0;
    virtual GA_Base* crossover(GA_Base*)=0;
    virtual int calcEvalution(void* aux)=0;
    /* -------- */
    
    bool operator<(const GA_Base* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const GA_Base* rhs){return this->_evalution > rhs->_evalution;}
    
    void setEvalution(int eval){_evalution = eval;}
    int getEvalution(){return _evalution;}
    
    void setProbability(double prob){probability = prob;}
    double getProbability(){return probability;}

protected:
    int _evalution;
    double probability;
};


#endif /* defined(__opencv_test__Structure__) */
