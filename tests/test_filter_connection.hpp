#ifndef TEST_FILTER_CONNECTION_HPP
#define TEST_FILTER_CONNECTION_HPP

#include "ff_cpg.hpp"

template<typename N, typename C>
class FilterPerceptron : public nn2::NN<N, C> {
public:
    FilterPerceptron(){
        this->set_nb_inputs(2);
        this->set_nb_outputs(1);

        this->full_connect(this->_inputs,this->_outputs,nn2::trait<typename N::weight_t>::zero());
        this->_outputs[0].no_bias();
    }
};

#endif //TEST_FILTER_CONNECTION_HPP
