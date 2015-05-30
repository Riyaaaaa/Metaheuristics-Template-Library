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
    
    T* mutation(){ static_cast<T*>(this)->mutation(); }
    T* inversion(){ static_cast<T*>(this)->inversion(); }
 
    T* crossOver(T*){ static_cast<T*>(this)->crossOver(); }
    
    int calcEvalution(U& aux){ static_cast<T*>(this)->calcEvalution(); }
    
    bool operator<(const T* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const T* rhs){return this->_evalution > rhs->_evalution;}

    enum class DNA_DISPLACEMENT_LISTS{
        MUTATION=0,
        INVERSION,
    };
    static const int NUMBER_OF_DISPLACEMENT=2;
    
    void setDistribution(std::array<double, NUMBER_OF_DISPLACEMENT> dist){_evolution_distribution = dist;}
    
    void setEvalution(int eval){_evalution = eval;}
    int getEvalution(){return _evalution;}
    
    void setProbability(double prob){probability = prob;}
    double getProbability(){return probability;}

protected:
    int _evalution;
    double probability;
    std::array<double, NUMBER_OF_DISPLACEMENT> _evolution_distribution;
};


#endif /* defined(__opencv_test__Structure__) */
