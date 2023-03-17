#include "test_filter_connection.hpp"

int main(int argc,char** argv){

    FilterPerceptron<nn2::ffcpg::neuron_t,nn2::ffcpg::connection_t> fp;
    fp.set_all_weights({1,1});

    for(int i = 0; i < 200; i++){
        std::vector<double> inputs = {1,static_cast<double>(i)*0.01f};
        fp.step(inputs,0);
        std::cout << inputs[0] << ";" << inputs[1] << ";" << fp.outf()[0] << std::endl;
    }

    return 0;
}

