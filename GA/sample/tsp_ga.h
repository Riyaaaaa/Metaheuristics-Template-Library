//
//  tsp.h
//  opencv_test
//
//  Created by Riya.Liel on 2015/04/24.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef __GA_tsp__
#define __GA_tsp__

#include"GABase.hpp"
#include"reference_list.h"
#include <vector>

class tsp_individual : public GA_Base{
public:
    using DNA = std::vector<int>;

    GA_Base* mutation()override;
    tsp_individual* pmx_cross_over(tsp_individual*);
    tsp_individual* cse_x_cross_over(tsp_individual*);
    GA_Base* crossover(GA_Base* source)override;
    
    int calcEvalution(void* aux)override;
    
    static DNA translateToDnaPhenotypicOrdinal(const DNA);
    static DNA translateToDnaPhenotypicTrait(const DNA);
    
    void setDNA(DNA& _dna){_phenotypic_trait=_dna;}
    void setDNA(DNA&& _dna){_phenotypic_trait=_dna;}
    
    //debug
    const DNA& getPhenotypic()const{return _phenotypic_trait;}
    
private:
    
    DNA _phenotypic_trait;
};

tsp_individual* makeTspIndividual(int number_of_city);

#endif /* defined(__opencv_test__tsp__) */
