//
//  SABase.hpp
//  Procon26
//
//  Created by Riya.Liel on 2015/06/05.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef Procon26_SABase_hpp
#define Procon26_SABase_hpp

#include<random>
#include<vector>

template<typename T,typename _AuxType,typename _StateType>
class SA_Base{ //Interface Template class for SA
public:
    
    SA_Base(_StateType state):_state(state){};
    
    using auxType = _AuxType;
    using stateType = _StateType;
    
    T& turnState(){ static_cast<T*>(this)->turnState(); }
    
    int calcEvalution(_AuxType& aux){ static_cast<T*>(this)->calcEvalution(); }
    
    bool operator<(const T* rhs){return this->_evalution < rhs->_evalution;}
    bool operator>(const T* rhs){return this->_evalution > rhs->_evalution;}
    
    void setEvalution(int eval){_evalution = eval;}
    int getEvalution(){return _evalution;}
    
    stateType getState(){return _state;}
    void setState(stateType state){_state = state;}
    
protected:
    stateType _state;
    int _evalution;
};


#endif
