#include "tsp_ga.h"
#include <random>

//#define RIYA_DEBUG

template<typename T>
std::vector<std::pair<std::pair<int,int>,std::pair<int,int>>>
enumrateSubtour(std::vector<T> source,std::vector<T> target);

tsp_individual* tsp_individual::cross_over(tsp_individual* source){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> distribution(0,static_cast<int>(_phenotypic_trait.size()-1));
    std::vector<int> pmx_map(_phenotypic_trait.size());
    
    int cross_point1 = distribution(mt)/2;
    int cross_point2 = distribution(mt)/2 + cross_point1;
    
    DNA source_dna = translateToDnaPhenotypicOrdinal(dynamic_cast<tsp_individual*>(source)->_phenotypic_trait);
    DNA parent_dna = translateToDnaPhenotypicOrdinal(this->_phenotypic_trait);
    
    tsp_individual* child = new tsp_individual;
    child->_phenotypic_trait = translateToDnaPhenotypicOrdinal(this->_phenotypic_trait);
    
    std::fill(pmx_map.begin(),pmx_map.end(),-1);
    
    for(int i=cross_point1;i<=cross_point2;i++) {
        auto it = std::find(pmx_map.begin(),pmx_map.end(),source_dna[i]);
        if(pmx_map.end() != it){
            if(pmx_map[child->_phenotypic_trait[i]]==-1)
                *it = child->_phenotypic_trait[i];
            else {
                *it = pmx_map[child->_phenotypic_trait[i]];
                pmx_map[child->_phenotypic_trait[i]] = -1;
            }
        }
        else if(pmx_map[child->_phenotypic_trait[i]] != -1){
            pmx_map[source_dna[i]] = pmx_map[child->_phenotypic_trait[i]];
            pmx_map[child->_phenotypic_trait[i]] = -1;
        }
        else{
            pmx_map[source_dna[i]] = child->_phenotypic_trait[i];
        }
        child->_phenotypic_trait[i] = source_dna[i];
    }
    
    for(int i=0;i<cross_point1;i++){
        if(pmx_map[child->_phenotypic_trait[i]] != -1){
            child->_phenotypic_trait[i] = pmx_map[child->_phenotypic_trait[i]];
        }
    }
    for(int i=cross_point2+1;i<child->_phenotypic_trait.size();i++){
        if(pmx_map[child->_phenotypic_trait[i]] != -1){
            child->_phenotypic_trait[i] = pmx_map[child->_phenotypic_trait[i]];
        }
    }
    
#ifdef RIYA_DEBUG
    for(int i=0;i<child->_phenotypic_trait.size();i++){
        std::cout << source_dna[i] << ' ';
    }
    std::cout << std::endl;
    for(int i=0;i<child->_phenotypic_trait.size();i++){
        std::cout << parent_dna[i] << ' ';
    }
    std::cout << std::endl;
    for(int i=0;i<child->_phenotypic_trait.size();i++){
        std::cout << child->_phenotypic_trait[i] << ' ';
    }
    std::cout << std::endl;
#endif
    
    child->_phenotypic_trait = translateToDnaPhenotypicTrait(child->_phenotypic_trait);
    
    return child;
}

tsp_individual* tsp_individual::cse_x_cross_over(tsp_individual* source){
    
    DNA source_dna = translateToDnaPhenotypicOrdinal(dynamic_cast<tsp_individual*>(source)->_phenotypic_trait);
    DNA parent_dna = translateToDnaPhenotypicOrdinal(this->_phenotypic_trait);
    
    tsp_individual* child = new tsp_individual;
    child->_phenotypic_trait = translateToDnaPhenotypicOrdinal(this->_phenotypic_trait);
    
    auto subtour_list = enumrateSubtour(parent_dna, source_dna);
    
    for(auto& subtour: subtour_list){
        if(subtour.first.first  > subtour.first.second )std::swap(subtour.first.first , subtour.first.second );
        if(subtour.second.first > subtour.second.second){
            std::swap(subtour.second.first, subtour.second.second);}
        std::copy(source_dna.begin() + subtour.second.first,
                  source_dna.begin() + subtour.second.second + 1,
                  child->_phenotypic_trait.begin() + subtour.first.first);
    }
    
    child->_phenotypic_trait = translateToDnaPhenotypicTrait(child->_phenotypic_trait);
    
    return child;
}

tsp_individual* tsp_individual::mutation(){
    tsp_individual* new_individual = new tsp_individual;
    DNA phenotypic_ordinal = translateToDnaPhenotypicOrdinal(_phenotypic_trait);
    
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> distribution(0,static_cast<int>(_phenotypic_trait.size()-1));
    
    int cross_point1 = distribution(mt);
    int cross_point2 = mt() % (_phenotypic_trait.size()-cross_point1);

    for(int i=cross_point1; i<=cross_point1 + (cross_point2-cross_point1)/2; i++){
        std::swap(phenotypic_ordinal[i],phenotypic_ordinal[cross_point1+cross_point2-i]);
    }
    
    new_individual->setDNA(translateToDnaPhenotypicTrait(phenotypic_ordinal));
    return new_individual;
}

tsp_individual::DNA tsp_individual::translateToDnaPhenotypicOrdinal(const DNA trait){
    DNA ordinal;
    DNA tmp;
    
    for(int i=0;i<trait.size();i++){
        tmp.push_back(i);
    }
    
    ordinal.resize(trait.size());
    
    for(int i=0;i<trait.size();i++){
        ordinal[i] = tmp[trait[i]];
        tmp.erase(tmp.begin() + trait[i]);
    }
    
    return ordinal;
}

tsp_individual::DNA tsp_individual::translateToDnaPhenotypicTrait(const DNA ordinal){
    DNA trait;
    DNA tmp;
    
    for(int i=0;i<ordinal.size();i++){
        tmp.push_back(i);
    }

    for(int i=0;i<ordinal.size();i++){
        for(int j=0;j<tmp.size();j++){
            if(ordinal[i] == tmp[j]){
                trait.push_back(j);
                tmp.erase(tmp.begin() + j);
                break;
            }
        }
    }

    
    return trait;
}

tsp_individual* makeTspIndividual(int number_of_city){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::vector<int> city_list;
    tsp_individual::DNA new_dna;
    tsp_individual* new_indivisual = new tsp_individual;
    int pos;
    
    for(int i=0;i<number_of_city;i++){
        city_list.push_back(i);
    }
    
    for(int i=0;i<number_of_city;i++){
        pos = mt() % city_list.size();
        new_dna.push_back(pos);
        city_list.erase(city_list.begin() + pos);
    }
    
    new_indivisual->setDNA(new_dna);
    
    return new_indivisual;
}

int tsp_individual::calcEvalution(std::vector<cv::Point>& city_list){
    int evalution=0,std_eval=0;
    
    
    for(int i=0;i<city_list.size();i++){
        std_eval += cv::norm(cv::Point(0,0)-city_list[i]);
    }
    
    
    DNA phenotypic_ordinal = translateToDnaPhenotypicOrdinal(_phenotypic_trait);
    
    for(int i=0;i<city_list.size()-1;i++){
        evalution+= cv::norm(city_list[phenotypic_ordinal[i]]-city_list[phenotypic_ordinal[i+1]]);
    }
    evalution+= cv::norm(city_list[phenotypic_ordinal.back()]-city_list[phenotypic_ordinal[0]]);

    _evalution = std_eval - evalution;
    return _evalution > 0 ? _evalution : 0;
}


void drowTspMaps(tsp_individual::DNA ordinal){
    
    
}

template<typename T>
std::vector<std::pair<std::pair<int,int>,std::pair<int,int>>>
enumrateSubtour(std::vector<T> source,std::vector<T> target){
    
    using myIterator = typename std::vector<T>::iterator;
    
    std::vector<std::pair<std::pair<int,int>,std::pair<int,int>>> subtour_list;
    
    bool flag=false;
    int la,lb,ra,rb;
    int left_a,left_b,right_a,right_b;
    int length=1;
    
    for(auto i=0; i<source.size();){
        la = i;
        lb = std::find(target.begin(),target.end(),source[la]) - target.begin();
        if(!flag){
            left_a = la;
            left_b = lb;
            right_b = lb;
            flag = true;
            i++;
        }
        else if (abs(right_b - lb)==1){
            right_a = la;
            right_b = lb;
            i++;
            length++;
        }
        else {
            if(length > 1){
                la = left_a;
                ra = right_a;
                lb = left_b;
                rb = right_b;
                subtour_list.push_back(make_pair( make_pair(left_a,right_a),
                                                 make_pair(left_b,right_b)
                                                 )
                                       );
            }
            flag = false; length = 1;
        }
        
    }
    return subtour_list;
}