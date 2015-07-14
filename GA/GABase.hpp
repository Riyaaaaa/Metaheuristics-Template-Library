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
#include <unordered_map>
#include <iostream>

template<typename T,typename U>
class GA_Base{ //Interface Template class for GA
public:
    using auxType = U;
    
    GA_Base(){ // static interface
        static_assert(!std::is_same< decltype(T().cross_over(nullptr)) , T*(T*) >::value , "crossover is not defined");
        static_assert(!std::is_same< decltype(T().mutation()) , T*(void) >::value , "mutation is not defined");
        static_assert(!std::is_same< decltype(T().calcEvalution(std::declval<U&>())) , int(U&) >::value , "calcEvalution is not defined"); //ill-formed
        //static_assert(!std::is_same< decltype(T().calcEvalution(std::declval<std::vector<int>&>())) , int(std::vector<int>&) >::value , "calcEvalution is not defined");
    }
    
    bool operator<(const T* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const T* rhs){return this->_evalution > rhs->_evalution;}
    
    void    setEvalution(int eval){_evalution = eval;}
    int     getEvalution(){return _evalution;}
    void    setProbability(double prob){probability = prob;}
    double  getProbability(){return probability;}

protected:
    int     _evalution;
    double  probability;
};



template<typename T,typename U>
class GA_Base_Multi{ //Interface Template class for GA
public:
    using auxType = U;
    
    /* Functions that you must define */
    
    int calcEvalution(U& aux){ return static_cast<T*>(this)->calcEvalution(); }
    
    /* -------- */
    
    bool operator<(const T* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const T* rhs){return this->_evalution > rhs->_evalution;}
    
    void setEvalution(int eval){_evalution = eval;}
    int getEvalution(){return _evalution;}
    
    void setProbability(double prob){probability = prob;}
    double getProbability(){return probability;}
    
    std::unordered_map< std::string,std::function<T* (T*)> > cross_over;
    std::unordered_map< std::string,std::function<T* (void)> > mutation;
    
protected:
    int _evalution;
    double probability;
};


#endif /* defined(__opencv_test__Structure__) */
