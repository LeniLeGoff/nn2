//| This file is a part of the nn2 module originally made for the sferes2 framework.
//| Adapted and modified to be used within the ARE framework by Léni Le Goff.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.




#ifndef _NN_CPG_HPP_
#define _NN_CPG_HPP_

#include "nn.hpp"
#include "neuron.hpp"
#include "af.hpp"
#include "pf.hpp"

typedef nn2::PfWSum<> pf_t;
typedef nn2::AfDirect<> af_t;
typedef nn2::Neuron<pf_t,af_t> neuron_t;
typedef nn2::Connection<> connection_t;

namespace nn2 {

  template<typename N, typename C>
/**
 * @brief An implementation of CPG network for the ARE framework
 */
class CPG : public NN<N, C> {
   public:
    typedef NN<N, C> nn_t;
    typedef typename nn_t::io_t io_t;
    typedef typename nn_t::vertex_desc_t vertex_desc_t;
    typedef typename nn_t::edge_desc_t edge_desc_t;
    typedef typename nn_t::adj_it_t adj_it_t;
    typedef typename nn_t::graph_t graph_t;
    typedef N neuron_t;
    typedef C conn_t;

    CPG(){}
    /**
    * @brief CPG
    * @param nb_inputs
    * @param nb_hidden
    * @param nb_outputs
    * @param joint_substrate a self-referencing connection vector.
    * -1 values indicate that they are attached to the head.
    * other values indicate the index that joint is attached to.
    * E.g. [-1,-1,-1,0,1,2] you have the first 3 joints attached to the head and the other connected as following:
    * 3->0  4->1  5->2 Example: {-1,-1,-1,0,1,2}
    *
    *
    *          3  0 |--------| 1  4
    *          --*--|        |__*--
    *               |        |
    *               |        |__*--
    *               |--------|2   5
    */
    CPG(size_t nb_inputs,
        size_t nb_hidden,
        size_t nb_outputs,
          std::vector<int> joint_substrate) {
      std::cout << "Contructing an CPG network" << std::endl;

      this->set_nb_inputs(nb_inputs + 1);
      this->set_nb_outputs(nb_outputs);

      // Create oscillators
      for(int i = 0; i < joint_substrate.size(); i++){
        auto neuron_a1 = this->add_neuron("A1"); //output of oscillator
        auto neuron_b1 = this->add_neuron("B1");
        this->create_oscillator_connection(neuron_a1, neuron_b1,
                                           trait<typename N::weight_t>::zero());
        _cpgs.push_back(neuron_a1);
        this->add_connection(neuron_a1,this->_outputs[i],
                trait<typename N::weight_t>::zero()); // connection to the outputs
      }

      // Connect oscillators with each others
      for(int i = 0; i < joint_substrate.size(); i++){
          if(joint_substrate.at(i) < 0){
              for(int j = 0; j < joint_substrate.size(); j++){
                  if(i==j)
                      continue;
                  if(joint_substrate.at(j) == joint_substrate.at(i)){
                      auto output = _cpgs.at(i);
                      auto input = _cpgs.at(j);
                      this->add_connection(output, input, 1.0,true);
                      joint_substrate[j]--;
                  }
              }
          }
          else {
            auto output = _cpgs.at(i);
            auto input = _cpgs.at(joint_substrate[i]);
            this->add_connection(output, input, 1.0,true);
          }
        }
    }
    unsigned get_nb_inputs() const {
      return this->_inputs.size() - 1;
    }
    void step(const std::vector<io_t>& in,double delta = 0.0) {
      assert(in.size() == this->get_nb_inputs());
      std::vector<io_t> inf = in;
      inf.push_back(1.0f);
      nn_t::_step_integrate(inf,delta);
    }

   protected:
    std::vector<vertex_desc_t> _cpgs; // CPG

  };
  namespace cpg {
    template<int NbInputs, int NbHidden, int Wheels, int PrimaryJoints, int SecondaryJoints>
    struct Count {
      const int nb_inputs = 0; // bias is an input
      const int nb_outputs =PrimaryJoints + SecondaryJoints;
      const int nb_hidden = 0;
      const int nb_weights = 1/2*PrimaryJoints*(PrimaryJoints - 1) + SecondaryJoints*2 + PrimaryJoints; // connection of the oscillators
      const int nb_biases = (PrimaryJoints+SecondaryJoints)*3;
    };
    struct count_t {
        int nb_inputs;
        int nb_outputs;
        int nb_hidden;
        int nb_weights;
        int nb_biases;
        count_t(int NbInputs,  int NbHidden,int Wheels,int PrimaryJoints, int SecondaryJoints){
            nb_inputs = 0;
            nb_outputs = PrimaryJoints + SecondaryJoints;;
            nb_hidden = 0;
            nb_weights = 1/2*PrimaryJoints*(PrimaryJoints - 1) + SecondaryJoints*2 + PrimaryJoints; // connection of the oscillators
            nb_biases = (PrimaryJoints+SecondaryJoints)*3;
        }

    };
  }
}

#endif
