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
    
    template<std::size_t surfaceSize>
    static bool statusScanning(std::array<Unit,surfaceSize> surface, std::array<double,surfaceSize> target){
        bool flag = true;
        double epsilon = 0.01;
        for(int i=0;i<surfaceSize; i++){
            flag = flag & ( fabs(surface[i].output(threshold()) - target[i]) < epsilon );
        }
        return flag;
    }
    
    template<std::size_t surfaceSize>
    static void inputting(std::array<Unit,surfaceSize+1>& surface, std::array<double,surfaceSize> input){
        for(int i=1;i<surfaceSize; i++){
            surface[i].setStatus(input[i-1]);
        }
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    class NNSolver {
    public:
        explicit NNSolver(double t_rate);
        
        typedef std::array<Unit , _First>    input_array;
        typedef std::array<Unit , _Last>     output_array;
        
        typedef typename mtl::make_tuple_array<Unit, std::tuple<>, _First+1, (Dims+1)..., _Last>::type network;
        
        network perceptron;
        
        typedef std::array<double,std::tuple_size<typename std::remove_reference< decltype(std::get<1>(perceptron)) >::type>::value > a;
        a _a;
        
        static constexpr int SURFACE_SIZE =  std::tuple_size<network>::value;
        const double TRAINIG_RATE;
        
        const output_array& solveAnswer(std::array<double, _First>);
        const output_array& training(std::vector<
                                     std::pair<
                                     std::array<double, std::tuple_size< input_array >::value>,
                                     std::array<double, std::tuple_size< output_array >::value>
                                     >
                                     >);
        template<typename... Args>
        const output_array& trainingImpl(std::array<double, std::tuple_size< input_array >::value>,
                                         std::array<double, std::tuple_size< output_array >::value>, Args&&... args);
        template<typename... Args>
        void makeTrainingPair(
                              std::vector<
                                            std::pair<
                                                    std::array<double, std::tuple_size< input_array >::value>,
                                                    std::array<double, std::tuple_size< output_array >::value>
                                                    >
                                        >&,
                              std::array<double, std::tuple_size< input_array >::value>,
                              std::array<double, std::tuple_size< output_array >::value>, Args&&... args);
        
        template<typename... Args>
        void makeTrainingPair(
                              std::vector<
                              std::pair<
                              std::array<double, std::tuple_size< input_array >::value>,
                              std::array<double, std::tuple_size< output_array >::value>
                              >
                              >&,
                              std::array<double, std::tuple_size< input_array >::value>,
                              std::array<double, std::tuple_size< output_array >::value>);
        
        void regulateWeight(std::array<double, std::tuple_size< input_array >::value>& input,
                            std::array<double, std::tuple_size< output_array >::value>& target);

        
    private:
        struct calcSurface;
        
        //input_array     _sensory;
        output_array    _response;
        
        /*
         template<typename Tuple, size_t... I>
         void NNapply_(Tuple& args, std::index_sequence<I...>)
         {
         calcSurface(std::get<I>(std::forward<Tuple>(args))...);
         }
         
         template<typename Tuple,
         typename Indices = std::make_index_sequence<std::tuple_size<Tuple>::value-1>>
         auto NNapply(Tuple& args)
         {
         NNapply_(args, Indices());
         }
         */
    };
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    NNSolver<_First , _Last , Dims...>::NNSolver(double t_rate):TRAINIG_RATE(t_rate){
        surfaceExecuteAll<0, SURFACE_SIZE-1>(perceptron, [](auto& surface){
            surface[0].setStatus(1);
            surface[0].isBias = true;
        });
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    const typename NNSolver<_First , _Last , Dims...>::output_array& NNSolver<_First , _Last , Dims...>::solveAnswer(std::array<double, _First>
                                                                                                                     sensory){
        
        inputting(std::get<0>(perceptron),sensory);
        
        mtl::forwardExecuteAll<0,SURFACE_SIZE-2>(perceptron, calcSurface());
        
        //_sensory = sensory;
        
        return std::get< SURFACE_SIZE -1 >(perceptron);
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    template<typename... Args>
    const typename NNSolver<_First , _Last , Dims...>::output_array&
    NNSolver<_First , _Last , Dims...>::trainingImpl(std::array<double, std::tuple_size< input_array >::value> input,
                                                    std::array<double, std::tuple_size< output_array >::value> target, Args&&... args){
        std::vector<std::pair<
        std::array<double, std::tuple_size< input_array >::value>,
        std::array<double, std::tuple_size< output_array >::value>
        >
        > training_list;
        makeTrainingPair(training_list,input,target,args...);
        
        return training(training_list);
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    template<typename... Args>
    void NNSolver<_First , _Last , Dims...>::makeTrainingPair(
                          std::vector<
                                        std::pair<
                                                    std::array<double, std::tuple_size< input_array >::value>,
                                                    std::array<double, std::tuple_size< output_array >::value>
                                                >
                                    >& list,
                          std::array<double, std::tuple_size< input_array >::value> input,
                          std::array<double, std::tuple_size< output_array >::value> target, Args&&... args){
        list.push_back(std::make_pair(input, target));
        makeTrainingPair(list, args...);
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    template<typename... Args>
    void NNSolver<_First , _Last , Dims...>::makeTrainingPair(
                                                              std::vector<
                                                              std::pair<
                                                              std::array<double, std::tuple_size< input_array >::value>,
                                                              std::array<double, std::tuple_size< output_array >::value>
                                                              >
                                                              >& list,
                                                              std::array<double, std::tuple_size< input_array >::value> input,
                                                              std::array<double, std::tuple_size< output_array >::value> target){
        list.push_back(std::make_pair(input, target));
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    const typename NNSolver<_First , _Last , Dims...>::output_array&
    NNSolver<_First , _Last , Dims...>::training(std::vector<
                                                            std::pair<
                                                                    std::array<double, std::tuple_size< input_array >::value>,
                                                                    std::array<double, std::tuple_size< output_array >::value>
                                                                    >
                                                            > training_list){
        const std::size_t TRAINIG_LIMITS = 1000;
        bool flag;
        
        for(int i=0; i<TRAINIG_LIMITS; i++){
            flag = true;
            for(auto& training_target: training_list){
                if(statusScanning(solveAnswer(training_target.first),training_target.second))continue;
                else {
                    flag = false;
                    regulateWeight(training_target.first, training_target.second);
                }
            }
            if(flag)break;
        }
        
        return std::get< SURFACE_SIZE-1 >(perceptron);
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    void NNSolver<_First , _Last , Dims...>::regulateWeight(std::array<double, std::tuple_size< input_array >::value>& input,
                                                       std::array<double, std::tuple_size< output_array >::value>& target){
        Backpropagation<network> _training;
        do{
            inputting(std::get<0>(perceptron), input);
            mtl::forwardExecuteAll<0, SURFACE_SIZE-2>(perceptron, calcSurface());
            _training.PropagationApply(std::move(perceptron),std::move(target),TRAINIG_RATE);
        }while(!statusScanning(std::get<SURFACE_SIZE-1>(perceptron),target));
        
    }
    
    template<std::size_t _First , std::size_t _Last , std::size_t... Dims>
    struct NNSolver<_First , _Last , Dims...>::calcSurface{
        template<std::size_t index>
        void operator()(network& perceptron){
            if(index != SURFACE_SIZE-1){
                double sum = sigma(std::get<index>(perceptron));
            
                for(auto& unit: std::get<index+1>(perceptron)){
                    unit.setStatus(sum);
                }
            }
        }
        
        
        template<std::size_t input_size>
        static double sigma(std::array<Unit,input_size>& input_surface){
            double sum=0;
            for(auto& unit: input_surface){
                sum += unit.output(sigmoid());
            }
            return sum;
        }
    };
    
}

#endif
