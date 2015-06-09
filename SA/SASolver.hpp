//
//  SASolver.h
//  Procon26
//
//  Created by Riya.Liel on 2015/06/04.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __GA_SASolver_
#define __GA_SASolver_

#include<cmath>

template<class _T,int _STime=1000,int _ETime=1,int _Schedule=99>
class SA_Solver{
public:
    SA_Solver(_T target):_target(target){}
    typename _T::stateType solveAnswer();
    void setAux(typename _T::auxType& aux){_aux = aux;}
private:
    
    double getProbability(int e1,int e2,double t){
        return e1 <= e2 ? 1.0 : exp( (e1-e2)/t );
    }
    typename _T::auxType _aux;
    _T _target;
};

template<class _T,int _STime,int _ETime,int _Schedule>
typename _T::stateType SA_Solver<_T,_STime,_ETime,_Schedule>::solveAnswer(){
    std::random_device _rnd;
    std::mt19937 _mt(_rnd());
    std::uniform_real_distribution<double> _distribution(0,1);

    double current_time=_STime;
    int best_eval=0, old_eval=0;
    typename _T::stateType old = _target.getState();
    typename _T::stateType best_state;
    int count=0;
    
    while(current_time >= _ETime){
        count++;
        _target.turnState();
        int next_eval = _target.calcEvalution(_aux);
        
        if(_distribution(_mt) <= getProbability(old_eval,next_eval,current_time)){
            if(best_eval < next_eval){
                best_state = _target.getState();
                best_eval = next_eval;
            }
            old_eval = next_eval;
        }
        else _target = old;
        
        current_time *= _Schedule/100.;
    }
    
    return best_state;
}

#endif
