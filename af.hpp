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

#ifndef _NN_AF_HPP_
#define _NN_AF_HPP_

#include "params.hpp"

// classic activation functions
namespace nn2 {
  template <typename P = params::Dummy>
  class Af {
   public:
    typedef P params_t;
    const params_t& get_params() const  {
      return _params;
    }
    params_t& get_params() {
      return _params;
    }
    void set_params(const params_t& params) {
      _params = params;
    }
    void init() {}
    Af() {}
   protected:
    params_t _params;
  };

  // -1 to +1 sigmoid
  template <typename P>
  struct AfTanh : public Af<P> {
    typedef P params_t;
    BOOST_STATIC_CONSTEXPR double lambda = 5.0f;
    AfTanh() {
      assert(trait<P>::size(this->_params) == 1);
    }
    float operator()(double p) const {
      return tanh(p * lambda + trait<P>::single_value(this->_params));
    }
   protected:
  };
  // -1 to +1 sigmoid
  template <typename P = double>
  struct AfTanhNoBias : public Af<P> {
    typedef params::Dummy params_t;
    BOOST_STATIC_CONSTEXPR float lambda = 5.0f;
    AfTanhNoBias() { }
    float operator()(double p) const {
      return tanh(p * lambda);
    }
  };



  template <typename P = double>
  struct AfSigmoid : public Af<> {
    typedef params::Dummy params_t;
    AfSigmoid() { }
    float operator()(double p) const {
      return 1.0 / (exp(-p) + 1);
    }
   protected:
  };

  template <typename P = std::vector<double>>
  struct AfSigmoidBias : public Af<P> {
    typedef P params_t;
    AfSigmoidBias() {
      assert(this->_params.size() == 2);
    }
    float operator()(double p) const {
      return 1.0 / (exp(-p * this->_params[0] + this->_params[1]) + 1);
    }
   protected:
  };

  template <typename P = std::vector<double>>
  struct AfSigmoidSigned : public Af<P> {
    typedef P params_t;
    AfSigmoidSigned() {
      //assert(this->_params.size() == 2);
    }
    float operator()(double p) const {
      return (1.0 / (exp(-p * this->_params[0] + this->_params[1]) + 1) - 0.5f)*2.f;
    }
   protected:
  };

  template <typename P = double>
  struct AfFilterSigmoidSigned : public Af<P> {
    typedef P params_t;
    AfFilterSigmoidSigned() {
      //assert(this->_params.size() == 2);
    }
    float operator()(std::vector<double> p) const {
      return (1.0 / (exp(-(1+p[1])*(p[0])) + 1) - 0.5f)*2.f;
    }
   protected:
  };



  // copy input to output
  // store an arbitrary parameter
  template<typename P = params::Dummy>
  struct AfDirect : public Af<P> {
    typedef P params_t;
    float operator()(double p) const {
      return p;
    }
  };

  // copy input to output
  template<typename T>
  struct AfDirectT : public Af<params::Dummy> {
    typedef params::Dummy params_t;
    T operator()(T p) const {
      return p;
    }
  };

}

#endif
