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
    static bool statusScanning(Layer layer, std::array<double, std::tuple_size<Layer>::value> target){
        bool flag = true;
        double epsilon = 0.01;
        for(int i=0; i<std::tuple_size<Layer>::value ; i++){
            flag = flag & ( fabs(threshold()(layer[i].getStatus()) - target[i]) < epsilon );
        }
        return flag;
    }
    
    template<class Layer>
    static void inputting(Layer layer, std::array<double, std::tuple_size<Layer>::value> input){
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
        
        template<typename... Args>
        const output_layer& trainingImpl(std::array<double, std::tuple_size< input_layer >::value>,
                                         std::array<double, std::tuple_size< output_layer >::value>, Args&&... args);
        template<typename... Args>
        void makeTrainingPair(
                              std::vector<
                                            std::pair<
                                                    std::array<double, std::tuple_size< input_layer >::value>,
                                                    std::array<double, std::tuple_size< output_layer >::value>
                                                    >
                                        >&,
                              std::array<double, std::tuple_size< input_layer >::value>,
                              std::array<double, std::tuple_size< output_layer >::value>, Args&&... args);
        
        template<typename... Args>
        void makeTrainingPair(
                              std::vector<
                              std::pair<
                              std::array<double, std::tuple_size< input_layer >::value>,
                              std::array<double, std::tuple_size< output_layer >::value>
                              >
                              >&,
                              std::array<double, std::tuple_size< input_layer >::value>,
                              std::array<double, std::tuple_size< output_layer >::value>);
        
    private:
        struct calcSurface;
        
        //input_layer     _sensory;
        output_layer    _response;
    };
    
    template<class NetworkStruct>
    NNSolver<NetworkStruct>::NNSolver(double t_rate):TRAINIG_RATE(t_rate){
        surfaceExecuteAll<0, LAYER_SIZE>(neural.network, [](auto& surface){
            for(int i=0; i<surface.size(); i++){
                surface[i].setStatus(1);
                surface[i].bias=0.5;
                std::fill(surface[i].weight.begin(),surface[i].weight.end(),0.5);
            }
        });
    }
    
    template<class NetworkStruct>
    auto NNSolver<NetworkStruct>::solveAnswer(std::array<double, std::tuple_size<input_layer>::value> sensory)
    ->const output_layer{
        
        inputting(std::get<0>(neural.network),sensory);
        
        mtl::forwardExecuteAll<0,LAYER_SIZE-2>(neural.network, calcSurface());
        
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
        const std::size_t TRAINIG_LIMITS = 1000;
        bool flag;
        
                      _TRAINING_OBJECT<typename NetworkStruct::structure> training_object;
                      
        for(int i=0; i<TRAINIG_LIMITS; i++){
            flag = true;
            for(auto& training_target: training_list){
                if(statusScanning(solveAnswer(training_target.first),training_target.second))continue;
                else {
                    flag = false;
                    regulateWeight(training_target.first, training_target.second, training_object);
                }
            }
            if(flag)break;
        }
        
        return std::get< LAYER_SIZE-1 >(neural);
    }
    
    template<class NetworkStruct>
    template<class _TRAINING_OBJECT>
    void NNSolver<NetworkStruct>::regulateWeight(std::array<double, std::tuple_size< input_layer >::value>& input,
                                                 std::array<double, std::tuple_size< output_layer >::value>& target,
                                                 _TRAINING_OBJECT& _training_algorithm){
        do{
            inputting(std::get<0>(neural), input);
            mtl::forwardExecuteAll<0, LAYER_SIZE-1>(neural.network, calcSurface());
            mtl::propagationTupleApply<LAYER_SIZE-1>(std::move(neural.network), std::move(_training_algorithm), std::move(target));
        }while(!statusScanning(std::get<LAYER_SIZE-1>(neural),target));
        
        
    }
    
    template<class NetworkStruct>
    struct NNSolver<NetworkStruct>::calcSurface{
        template<std::size_t index>
        void operator()(typename NetworkStruct::structure& network){
            if(index != LAYER_SIZE-1){
                double sum = sigma(std::get<index>(network));
            
                for(auto& unit: std::get<index+1>(network)){
                    unit.setStatus(sum);
                }
            }
        }
        
        
        template<class Layer>
        static double sigma(Layer& input_layer){
            double sum=0;
            for(auto& unit: input_layer){
                sum += unit.output(sigmoid());
            }
            return sum;
        }
    };
    
}

#endif
