#include <iostream>
#include <fstream>
#include <random>
#include "tsp_sa.h"
#include "SASolver.hpp"

#ifndef __tsp_sa__
#define __tsp_sa__

int test_sa(){
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> distiribution(0,500);
    std::vector<cv::Point> city_list;
    std::vector<int> city_hash,ans;
    const int city_size = 100;
    int pos;
    
    ifstream ifs("./../../../../../MTL_Development/cities.txt");
    
    for(int i=0;i<city_size;i++){
        city_list.push_back(cv::Point());
        ifs >> city_list.back().x >> city_list.back().y;
    }
    
    for(int i=0;i<city_size;i++){
        city_hash.push_back(i);
    }
    
    for(int i=0;i<city_size;i++){
        pos = mt() % city_hash.size();
        ans.push_back(city_hash[pos]);
        city_hash.erase(city_hash.begin() + pos);
    }
    
    tsp_annealing target(ans);
    
    SA_Solver<tsp_annealing> solver( target );
    solver.setAux(city_list);
    
    target = solver.solveAnswer();
        
        cv::Mat drawImage(500, 500, CV_8UC3, cv::Scalar(0));
        for(cv::Point p : city_list)
        {
            cv::line(drawImage, p, p, cv::Scalar(255, 0, 0),5);
        }
        
        for(int i=0;i<ans.size();i++)
        {
            std::cout << ans[i] << ' ';
        }
        std::cout << std::endl;
        
        int eval=0;
        for(int i=0;i<ans.size()-1;i++)
        {
            cv::line(drawImage, city_list[ans[i]], city_list[ans[i+1]], cv::Scalar(255, 0, 0),1);
            eval += cv::norm(city_list[ans[i]]-city_list[ans[i+1]]);
        }
        eval += cv::norm(city_list[ans.front()]-city_list[ans.back()]);
        std::cout << eval << std::endl;
        
        cv::line(drawImage, city_list[ans.back()], city_list[ans.front()], cv::Scalar(255, 0, 0),1);
        cv::namedWindow("test", CV_WINDOW_AUTOSIZE);
        cv::imshow("test", drawImage);
        while (true)
        {
            int key = cv::waitKey(1);
            if (key == 0x1b) break;
        }
    return 0;
}

#endif
