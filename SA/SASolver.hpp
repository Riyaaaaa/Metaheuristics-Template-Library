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
#include "SABase.hpp"

/* Prototype Definition */
/* Setting default template for arguments for cooling schedule */
/* default argments: _Stime : initial of time / _Etime : end of time / _Schedule : percent for cooling schedule */
template<class _T,int _STime=100,int _ETime=1,int _Schedule=90>
class _SA_Solver;

/* To check Class whether or not it extend SA_Base Class*/
template<class _T,int _STime=100,int _ETime=1,int _Schedule=90>
using SA_Solver = typename std::enable_if< std::is_base_of<SA_Base<_T,typename _T::auxType,typename _T::stateType>,
                                                            _T
                                                          >::value,
                                            _SA_Solver<_T,_STime,_ETime,_Schedule>
                                          > ::type;

template<class _T,int _STime,int _ETime,int _Schedule>
class _SA_Solver{
public:
    _SA_Solver(_T target):_target(target){}
    _T solveAnswer();
    void setAux(typename _T::auxType& aux){_aux = aux;}
private:
    
    double getProbability(int e1,int e2,double t){
        return e1 <= e2 ? 1.0 : exp( (e1-e2)/t );
    }
    typename _T::auxType _aux;
    _T _target;
};

template<class _T,int _STime,int _ETime,int _Schedule>
_T _SA_Solver<_T,_STime,_ETime,_Schedule>::solveAnswer(){
    std::random_device _rnd;
    std::mt19937 _mt(_rnd());
    std::uniform_real_distribution<double> _distribution(0,1);

    double current_time=_STime;
    
    _target.initState(_aux);
    
    _T old = _target, best(_target.getState());
    typename _T::stateType best_state;
    int  old_eval=_target.calcEvalution(_aux) , best_eval=0;
    
    while(current_time >= _ETime){
        old = _target;
        _target.turnState(_aux);
        int next_eval = _target.calcEvalution(_aux);
        
        if(_distribution(_mt) <= getProbability(old_eval,next_eval,current_time)){
            if(best_eval < next_eval){
                best = _target;
                best_eval = next_eval;
            }
            old_eval = next_eval;
        }
        else _target = old;
        
        current_time *= _Schedule/100.;
    }
    _target = best;
    
    return _target;
}

#endif
