//
//  GASolver.h
//  Procon26
//
//  Created by Riya.Liel on 2015/05/20.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __GA_GA_Solver__
#define __GA_GA_Solver__

#include "GABase.hpp"
#include <random>
#include <algorithm>
#include "GAAlgorithm.hpp"

/* Prototype Definition */
/* Setting default template for arguments algorithm function objects */
/* default argments: Individual selector : Roulette Select Algorithm.
                     Evalution scaler : Power Scaling Algorithm. */
template <  class _Individual,int _NumberOfIndividual,
            class _Selector = RouletteSelect<_Individual>,
            class _Scaler = PowerScaling<_Individual> >
class _GA_Solver;

/* To check Individual Class whether or not it extend GA_Base Class*/

template<class _Individual,int _NumberOfIndividual>
using GA_Solver = typename std::enable_if<
                            std::is_base_of<GA_Base<_Individual,typename _Individual::auxType>,_Individual>::value,
                            _GA_Solver<_Individual,_NumberOfIndividual> > ::type;


//Don't use insead of using template alias GA_Solver
template< class _Individual,int _NumberOfIndividual,class _Selector,class _Scaler>
class _GA_Solver{
public:
    _GA_Solver(std::vector<_Individual*> solve_targets);
    _Individual* solveAnswer(int);
    
    void populationSettings();

    const double PROBABILITY_OF_MUTATION=0.01;
    const double PROBABILITY_OF_INVERSION=0.01;
    
    const std::vector<_Individual*>& getPopulation(){return _population;}
    
    void setAux(typename _Individual::auxType& aux){_aux = aux;}
private:
    typename _Individual::auxType _aux;
    
    _Selector _selector;
    _Scaler _scaler;
    
    std::random_device _rnd;
    std::mt19937 _mt;
    std::uniform_real_distribution<double> _distribution;
    std::vector<_Individual*> _population;
};

template<typename _Individual,int _NumberOfIndividual,class _Selector,class _Scaler>
_GA_Solver<_Individual,_NumberOfIndividual,_Selector,_Scaler>::_GA_Solver(std::vector<_Individual*> solve_targets){
    _population = solve_targets;
    _mt = std::mt19937(_rnd());
    _distribution = std::uniform_real_distribution<double>(0,1);
}

template<class _Individual,int _NumberOfIndividual,class _Selector,class _Scaler>
_Individual* _GA_Solver<_Individual,_NumberOfIndividual,_Selector,_Scaler>::solveAnswer(int max_age){
    _Individual* answer;
    std::vector<_Individual*> new_poplation;
    
    populationSettings();
    
    for(int i=0;i<max_age;i++){
        new_poplation.push_back(_population.front());
        
        while(new_poplation.size() < _NumberOfIndividual){
            _Individual *father = _selector(std::move(_population)),
            *mother = _selector(std::move(_population));
            new_poplation.push_back(father->cross_over(mother));
            if(_distribution(_mt) < PROBABILITY_OF_MUTATION){
                new_poplation.push_back(_population.back()->mutation());
            }
            else if(_distribution(_mt) < PROBABILITY_OF_INVERSION){
                new_poplation.push_back(_population.back()->inversion());
            }
        }
        
        _population.erase(_population.begin());
        for(auto ptr: _population){
            delete ptr;
        }
        
        _population = new_poplation;
        new_poplation.clear();
        
        try{
            populationSettings();
        }
        catch(std::string str){
            std::cout << str << std::endl;
            break;
        }
    }
    
    answer = _population.front();
    return answer;
}

template<class _Individual,int _NumberOfIndividual,class _Selector,class _Scaler>
void _GA_Solver<_Individual,_NumberOfIndividual,_Selector,_Scaler>::populationSettings(){
    for(_Individual* individual : _population){
        individual->setEvalution(individual->calcEvalution(_aux));
    }
    std::sort(_population.begin(), _population.end(), [](_Individual* lhs,_Individual* rhs){return lhs->getEvalution() > rhs->getEvalution();}); //descending sort
    
    _scaler(_population,2);
    
    double sum=0;
    double sumProbablity=0;
    
    for(_Individual* individual : _population){
        sum+=individual->getEvalution();
    }
    
    if(sum == 0)throw "all individuals are same DNA.";
    
    for(_Individual* individual : _population){
        individual->setProbability(individual->getEvalution()/sum + sumProbablity);
        sumProbablity = individual->getProbability();
    }
}

#endif /* defined(__Procon26__GASolver__) */
