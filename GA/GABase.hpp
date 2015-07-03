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

template<typename T,typename U>
class GA_Base{ //Interface Template class for GA
public:
    using auxType = U;
    
    /* Functions that you must define */
    
    T* mutation(){ return static_cast<T*>(this)->mutation(); }
    T* crossOver(T* t){ return static_cast<T*>(this)->crossOver(t); }
    int calcEvalution(U& aux){ return static_cast<T*>(this)->calcEvalution(); }
    
    /* -------- */
    
    bool operator<(const T* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const T* rhs){return this->_evalution > rhs->_evalution;}
    
    void setEvalution(int eval){_evalution = eval;}
    int getEvalution(){return _evalution;}
    
    void setProbability(double prob){probability = prob;}
    double getProbability(){return probability;}

protected:
    int _evalution;
    double probability;
};


#endif /* defined(__opencv_test__Structure__) */
