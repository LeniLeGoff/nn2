#include <iostream>
#include <opencv2/opencv.hpp>
#include "cppn.hpp"

std::mt19937 nn2::rgen_t::gen;
struct params{
    static float _rate_add_neuron;
    static float _rate_del_neuron;
    static float _rate_add_conn;
    static float _rate_del_conn;
    static float _rate_change_conn;

    static size_t _min_nb_neurons;
    static size_t _max_nb_neurons;
    static size_t _min_nb_conns;
    static size_t _max_nb_conns;
};

float params::_rate_add_neuron = 1.0;
float params::_rate_del_neuron = 0.1;
float params::_rate_add_conn = 1.0;
float params::_rate_del_conn = 0.1;
float params::_rate_change_conn = 0.5;

size_t params::_min_nb_neurons = 2;
size_t params::_max_nb_neurons = 30;
size_t params::_min_nb_conns = 1;
size_t params::_max_nb_conns = 100;

int main(int argc,char** argv){

    nn2::rgen_t::gen.seed(time(0));


    nn2::CPPN<params> cppn;

    cppn.random();
    cppn.init();

    cv::Mat image(400,400,CV_8UC1);
    std::vector<double> outs;
    for(int k = 0; k < 10; k++){
        std::cout << "start" << std::endl;
        for(int i = 0; i < 400; i++){
            for(int j = 0; j < 400; j++){
                float u = static_cast<float>(i)/200. - 1;
                float v = static_cast<float>(j)/200. - 1;
                cppn.step({12*u,12*v});
                outs = cppn.outf();
                image.at<uchar>(i,j,0) = 128*outs[0]+127;
            }
        }
        std::cout << "finish" << std::endl;
        cv::imshow("Random CPPN Generated Image",image);
        cv::waitKey(0);
        cppn.mutate();
    }
    return 0;
}
