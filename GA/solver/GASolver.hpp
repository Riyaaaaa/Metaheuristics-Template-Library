//
//  GASolver.h
//  Procon26
//
//  Created by Riya.Liel on 2015/05/20.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __Procon26__GASolver__
#define __Procon26__GASolver__

#include "GABase.hpp"
#include <random>
#include <algorithm>
#include <iostream>
#include <type_traits>

template<class T,int N_I>
class _GA_Solver;

template<bool is_base,class T,int N_I>
struct DUMMY;

template<class T,int N_I>
struct DUMMY<true,T,N_I>{
    using type = _GA_Solver<T,N_I>;
};

template<class T,int N_I>
using GA_Solver = typename DUMMY<std::is_base_of<GA_Base<T,typename T::auxType>,T>::value,T,N_I>::type;

template<class T,int N_I>
class _GA_Solver{
public:
    _GA_Solver(std::vector<T*> solve_targets);
    T* select();
    T* solveAnswer(int);
    
    void populationSettings();
    
    const int NUMBER_OF_INDIVIDUAL=N_I;
    const double PROBABILITY_OF_MUTATION=0.01;
    const double PROBABILITY_OF_INVERSION=0.01;
    
    void setAux(typename T::auxType& aux){_aux = aux;}
private:
    typename T::auxType _aux;
    std::random_device _rnd;
    std::mt19937 _mt;
    std::uniform_real_distribution<double> _distribution;
    std::vector<T*> _population;
};

template<typename T,int N_I>
_GA_Solver<T,N_I>::_GA_Solver(std::vector<T*> solve_targets){
    _population = solve_targets;
    _mt = std::mt19937(_rnd());
    _distribution = std::uniform_real_distribution<double>(0,1);
}

template<typename T,int N_I>
T* _GA_Solver<T,N_I>::select(){
    T* individual=nullptr;
    
    double rnd = _distribution(_mt);
    
    for(int i=0;i<_population.size();i++){
        if(_population[i]->getProbability() > rnd){
            individual = _population[i];
            //std::cout << "selected" << i << std::endl;
            break;
        }
    }
    if(!individual)throw "individual is null";
    return individual;
}

template<typename T,int N_I>
T* _GA_Solver<T,N_I>::solveAnswer(int max_age){
    T* answer;
    std::vector<T*> new_poplation;
    
    for(int i=0;i<max_age;i++){
        populationSettings();
        new_poplation.push_back(_population.front());
        
        while(new_poplation.size() < NUMBER_OF_INDIVIDUAL){
            T* father = select(),*mother = select(),*target = select();
            new_poplation.push_back(father->cross_over(mother));
            if(_distribution(_mt) < PROBABILITY_OF_MUTATION){
                new_poplation.push_back(target->mutation());
            }
            else if(_distribution(_mt) < PROBABILITY_OF_INVERSION){
                new_poplation.push_back(target->inversion());
            }
        }
        
        _population.erase(_population.begin());
        for(auto ptr: _population){
            delete ptr;
        }
        
        _population = new_poplation;
        new_poplation.clear();
    }
    
    populationSettings();
    answer = _population.front();
    return answer;
}

template<class T,int N_I>
void _GA_Solver<T,N_I>::populationSettings(){
    for(T* individual : _population){
        individual->calcEvalution(_aux);
    }
    std::sort(_population.begin(), _population.end(), [](T* lhs,T* rhs){return lhs->getEvalution() > rhs->getEvalution();}); //descending sort
    
    double sum=0;
    double sumProbablity=0;
    
    for(T* individual : _population){
        individual->setEvalution(individual->getEvalution() * individual->getEvalution());
        sum+=individual->getEvalution();
    }
    
    if(sum == 0)throw "sum is 0";
    
    for(T* individual : _population){
        individual->setProbability(individual->getEvalution()/sum + sumProbablity);
        sumProbablity = individual->getProbability();
    }
}

#endif /* defined(__Procon26__GASolver__) */
