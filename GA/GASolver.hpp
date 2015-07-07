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

class GA_Solver{
public:
    GA_Solver(std::vector<GA_Base*> solve_targets);
    GA_Base* solveAnswer(int);
    
    void populationSettings();

    static constexpr double PROBABILITY_OF_CROSSOVER=1.0;
    static constexpr double PROBABILITY_OF_MUTATION=0.01;
    
    const std::vector<GA_Base*>& getPopulation(){return _population;}
    
    void setAux(void* aux){_aux = aux;}
private:
    void* _aux;
    
    RouletteSelect<GA_Base> _selector;
    PowerScaling<GA_Base> _scaler;
    
    std::random_device _rnd;
    std::mt19937 _mt;
    std::uniform_real_distribution<double> _distribution;
    std::vector<GA_Base*> _population;
};


GA_Solver::GA_Solver(std::vector<GA_Base*> solve_targets):
    _population(solve_targets),
    _mt(std::mt19937(_rnd())),
    _distribution(std::uniform_real_distribution<double>(0,1))
{}

GA_Base* GA_Solver::solveAnswer(int max_age){
    GA_Base* answer;
    std::vector<GA_Base*> new_poplation;
    
    populationSettings();
    
    for(int i=0;i<max_age;i++){
        new_poplation.push_back(_population.front());
        
        while(new_poplation.size() < _population.size()){
            GA_Base *father = _selector(std::move(_population)),
            *mother = _selector(std::move(_population));
            
            new_poplation.push_back(father->crossover(mother));
            
            if(_distribution(_mt) < PROBABILITY_OF_MUTATION){
                new_poplation.push_back(_population.back()->mutation());
            }
            
            new_poplation.back()->setEvalution(new_poplation.back()->calcEvalution(_aux));
            
            /*if(new_poplation.end() == std::find_if(new_poplation.begin(),new_poplation.end(),
                                                   [&](_Individual* rhs){
                                                       return new_poplation.back()->getEvalution() == rhs->getEvalution();
                                                   }
                                                   )
               )
                new_poplation.pop_back();*/
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

void GA_Solver::populationSettings(){
    for(GA_Base* individual : _population){
        individual->setEvalution(individual->calcEvalution(_aux));
    }
    std::sort(_population.begin(), _population.end(), [](GA_Base* lhs,GA_Base* rhs){return lhs->getEvalution() > rhs->getEvalution();}); //descending sort
    
    _scaler(_population,2);
    
    double sum=0;
    double sumProbablity=0;
    
    for(GA_Base* individual : _population){
        sum+=individual->getEvalution();
    }
    
    if(sum == 0)throw "all individuals are same DNA.";
    
    for(GA_Base* individual : _population){
        individual->setProbability(individual->getEvalution()/sum + sumProbablity);
        sumProbablity = individual->getProbability();
    }
}

#endif /* defined(__Procon26__GASolver__) */
