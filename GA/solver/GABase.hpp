//
//  Structure.h
//  opencv_test
//
//  Created by Riya.Liel on 2015/04/24.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __opencv_test__Structure__
#define __opencv_test__Structure__

#include <vector>
#include <array>

template<typename U,typename T>
class GA_Base{ //Interface class for GA
public:
    using auxType = T;
    
    virtual U* mutation()=0;
    virtual U* inversion()=0;
    //virtual GA_Base* select(std::vector<GA_Base*>&)=0;
    virtual U* cross_over(U*)=0;
    virtual int calcEvalution(T& aux)=0;
    
    bool operator<(const U* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const U* rhs){return this->_evalution > rhs->_evalution;}

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
    
    virtual ~GA_Base()=default;
    
protected:
    int _evalution;
    double probability;
    std::array<double, NUMBER_OF_DISPLACEMENT> _evolution_distribution;
};


#endif /* defined(__opencv_test__Structure__) */
