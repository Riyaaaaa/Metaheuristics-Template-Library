#include<iostream>
#include <random>
#include"tsp_ga.h"
#include "GASolver.hpp"

#ifndef __tsp_ga__
#define __tsp_ga__

void test_ga(){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> distiribution(0,500);
    std::vector<cv::Point> city_list;
    std::vector<tsp_individual*> population;
    int count=1;
    const int max_age=1000,city_size=100,individual_size=city_size;
    
    
     for(int i=0;i<city_size;i++){
     city_list.push_back(cv::Point(distiribution(mt),distiribution(mt)));
     }
     
    
    /*
    for(int i=0; i<city_size; i++){
        city_list.push_back(cv::Point(250+200*cos(i*(2*3.14)/city_size),250+200*sin(i*(2*3.14)/city_size)));
    }
     */
    
    for(int i=0;i<individual_size;i++){
        population.push_back(makeTspIndividual(city_size));
    }
    
    std::cout << cv::norm(city_list[0]-city_list[1]) << std::endl;
    
    GA_Solver<tsp_individual,individual_size> solver(population);
    solver.setAux(city_list);
    
    auto start = std::chrono::system_clock::now();
    
    solver.solveAnswer(max_age);
    
    auto end = std::chrono::system_clock::now();
    
    auto diff = end - start;
    std::cout << "elapsed time = "
    << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
    << " msec."
    << std::endl;
    
    /*
    solver.populationSettings();
    tsp_individual::DNA root = tsp_individual::translateToDnaPhenotypicOrdinal(solver.getPopulation().front()->getPhenotypic());
    while(true){
        std::cout << count * max_age << "世代目" << std::endl;
        
        cv::Mat drawImage(500, 500, CV_8UC3, cv::Scalar(0));
        for(cv::Point p : city_list)
        {
            cv::line(drawImage, p, p, cv::Scalar(255, 0, 0),5);
        }
        
        for(int i=0;i<root.size();i++)
        {
            std::cout << root[i] << ' ';
        }
        std::cout << std::endl;
        
        int eval=0;
        for(int i=0;i<root.size()-1;i++)
        {
            cv::line(drawImage, city_list[root[i]], city_list[root[i+1]], cv::Scalar(255, 0, 0),1);
            eval += cv::norm(city_list[root[i]]-city_list[root[i+1]]);
        }
        eval += cv::norm(city_list[root.front()]-city_list[root.back()]);
        std::cout << eval << std::endl;
        
        cv::line(drawImage, city_list[root.back()], city_list[root.front()], cv::Scalar(255, 0, 0),1);
        cv::namedWindow("test", CV_WINDOW_AUTOSIZE);
        cv::imshow("test", drawImage);
        while (true)
        {
            int key = cv::waitKey(1);
            if (key == 0x1b) break;
        }
        count++;
        
        tsp_individual* answer = solver.solveAnswer(max_age);
        root =  answer->translateToDnaPhenotypicOrdinal(answer->getPhenotypic());
    }
     */
}

#endif
