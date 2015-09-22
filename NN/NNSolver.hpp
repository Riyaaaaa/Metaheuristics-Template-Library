//
//  NNSolver.hpp
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/13.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_NNSolver_hpp
#define MTL_Development_NNSolver_hpp

#include<iostream>
#include<functional>
#include"Algorithm.hpp"
#include"NNBase.hpp"
#include"Utility.hpp"

namespace mtl{
    
    template<class Layer>
    static double statusScanning(Layer layer, std::array<double, std::tuple_size<Layer>::value> target){
        double RMSerror=0.0;
        for(int i=0; i<std::tuple_size<Layer>::value ; i++){
            RMSerror += fabs(layer[i].getStatus() - target[i]);
            std::cout << i+1 << "th units output " << layer[i].getStatus() << ", target value= " << target[i] << std::endl;
        }
        return RMSerror;
    }
    
    template<class Layer>
    static void inputting(Layer& layer, std::array<double, std::tuple_size<Layer>::value> input){
        for(int i=0; i<std::tuple_size<Layer>::value; i++){
            layer[i].setStatus(input[i]);
        }
    }
    
    template<class NetworkStruct>
    class NNSolver {
    public:
        explicit NNSolver(double t_rate);
        
        NetworkStruct neural;
        static constexpr std::size_t LAYER_SIZE = NetworkStruct::LAYER_SIZE;
        
        typedef typename NetworkStruct::template layer_type<LAYER_SIZE-1>   output_layer;
        typedef typename NetworkStruct::template layer_type<0>              input_layer;
        
        const double TRAINIG_RATE;
        
        auto solveAnswer(std::array<double, std::tuple_size<input_layer>::value>)
        ->const output_layer;
        
        template<template<class>class _TRAINING_OBJECT>
        auto training(std::vector<
                 std::pair<
                      std::array<double, std::tuple_size<input_layer>::value>,
                      std::array<double, std::tuple_size<output_layer>::value>
                 >
                 > training_list)
        ->const typename NetworkStruct::template layer_type<LAYER_SIZE-1>&;
        
        template<class _TRAINING_OBJECT>
        void regulateWeight(std::array<double, std::tuple_size< input_layer >::value>& input,
                            std::array<double, std::tuple_size< output_layer >::value>& target,
                            _TRAINING_OBJECT& _training_algorithm);
        
    private:
        struct calcSurface;
        
        //input_layer     _sensory;
        output_layer    _response;
    };
    
    template<class NetworkStruct>
    NNSolver<NetworkStruct>::NNSolver(double t_rate):TRAINIG_RATE(t_rate){
        surfaceExecuteAll<0, LAYER_SIZE>(neural.network, [](auto& surface){
            std::random_device rnd;
            std::mt19937 mt(rnd());
            std::uniform_real_distribution<double> distribution(-1,1);
            for(int i=0; i<surface.size(); i++){
                surface[i].setStatus(distribution(mt));
                surface[i].bias=(distribution(mt));
                std::fill(surface[i].weight.begin(),surface[i].weight.end(),distribution(mt));
            }
        });
    }
    
    template<class NetworkStruct>
    auto NNSolver<NetworkStruct>::solveAnswer(std::array<double, std::tuple_size<input_layer>::value> sensory)
    ->const output_layer{
        
        inputting(std::get<0>(neural.network),sensory);
        
        mtl::forwardExecuteAll<0,LAYER_SIZE-1>(neural.network, calcSurface());
        
        //_sensory = sensory;
        
        return std::get< LAYER_SIZE -1 >(neural.network);
    }
    
                  /*
    template<class NetworkStruct>
    template<typename... Args>
                  const NetworkStruct::template layer_type<LAYER_SIZE-1>&
    NNSolver<NetworkStruct>::trainingImpl(std::array<double, std::tuple_size< input_layer >::value> input,
                                                    std::array<double, std::tuple_size< output_layer >::value> target, Args&&... args){
        std::vector<std::pair<
        std::array<double, std::tuple_size< input_layer >::value>,
        std::array<double, std::tuple_size< output_layer >::value>
        >
        > training_list;
        makeTrainingPair(training_list,input,target,args...);
        
        return training(training_list);
    }
    
    template<class NetworkStruct>
    template<typename... Args>
    void NNSolver<NetworkStruct>::makeTrainingPair(
                          std::vector<
                                        std::pair<
                                                    std::array<double, std::tuple_size< input_layer >::value>,
                                                    std::array<double, std::tuple_size< output_layer >::value>
                                                >
                                    >& list,
                          std::array<double, std::tuple_size< input_layer >::value> input,
                          std::array<double, std::tuple_size< output_layer >::value> target, Args&&... args){
        list.push_back(std::make_pair(input, target));
        makeTrainingPair(list, args...);
    }
    
    template<class NetworkStruct>
    template<typename... Args>
    void NNSolver<NetworkStruct>::makeTrainingPair(
                                                              std::vector<
                                                              std::pair<
                                                              std::array<double, std::tuple_size< input_layer >::value>,
                                                              std::array<double, std::tuple_size< output_layer >::value>
                                                              >
                                                              >& list,
                                                              std::array<double, std::tuple_size< input_layer >::value> input,
                                                              std::array<double, std::tuple_size< output_layer >::value> target){
        list.push_back(std::make_pair(input, target));
    }
                   */
    
    template<class NetworkStruct>
    template<template<class>class _TRAINING_OBJECT>
    auto NNSolver<NetworkStruct>::training(std::vector<
                                                            std::pair<
                                                                    std::array<double, std::tuple_size<input_layer>::value>,
                                                                    std::array<double, std::tuple_size<output_layer>::value>
                                                                    >
                                                            > training_list)
                  ->const typename NetworkStruct::template layer_type<LAYER_SIZE-1>&{
        const std::size_t TRAINIG_LIMITS = 2000;
        
                      _TRAINING_OBJECT<typename NetworkStruct::structure> training_object;
                      double RMSerror = 0.0, best = 1e6;
                      
        for(int i=0; i<TRAINIG_LIMITS; i++){
                        std::random_shuffle(training_list.begin(), training_list.end());
            for(auto& training_target: training_list){
                regulateWeight(training_target.first, training_target.second, training_object);
                RMSerror += statusScanning(solveAnswer(training_target.first),training_target.second);
            }
            std::cout << "RMSerror = " << RMSerror << std::endl;
            if(best > RMSerror){ best = RMSerror; }
            RMSerror = 0.0;
            //else break;
        }
            std::cout << "best value = " << best << std::endl;
        return std::get< LAYER_SIZE-1 >(neural);
    }
    
    template<class NetworkStruct>
    template<class _TRAINING_OBJECT>
    void NNSolver<NetworkStruct>::regulateWeight(std::array<double, std::tuple_size< input_layer >::value>& input,
                                                 std::array<double, std::tuple_size< output_layer >::value>& target,
                                                 _TRAINING_OBJECT& _training_algorithm){
            inputting(std::get<0>(neural), input);
            mtl::forwardExecuteAll<0, LAYER_SIZE-1>(neural.network, calcSurface());
            mtl::propagationTupleApply<LAYER_SIZE-1>(std::move(neural.network), std::move(_training_algorithm), std::move(target));
        
    }
    
    template<class NetworkStruct>
    struct NNSolver<NetworkStruct>::calcSurface{
        template<std::size_t index>
        void operator()(typename NetworkStruct::structure& network){
            auto& layer = std::get<index+1>(network);
            if(index != LAYER_SIZE-1){
                for(int i=0; i<layer.size(); i++){
                    double sum = sigma(std::get<index>(network),i);
                    //layer[i].setStatus(sigmoid()(sum+layer[i].bias));
                    layer[i].setStatus(tanh(sum+layer[i].bias));
                }
            }
        }
        
        
        template<class Layer>
        static double sigma(Layer& input_layer,int unitid){
            double sum=0;
            for(auto& unit: input_layer){
                sum += unit.getStatus() * unit.weight[unitid] + unit.bias;
            }
            return sum;
        }
    };
    
}

#endif
