#ifndef CPPN_HPP_
#define CPPN_HPP_

#include "nn.hpp"
#include "evo_float.hpp"
#include "random.hpp"
#include <random>

namespace nn2{

namespace cppn{

typedef EvoFloat<1,evo_float::default_params> evo_float_t;

struct Params : params::Dummy{
    int type;
    evo_float_t p0;
    evo_float_t p1;

    void random(){
        type = std::uniform_int_distribution<>(0,3)(rgen_t::gen);
        p0.random();
        p1.random();
    }
    void mutate(){
        p0.mutate();
        p1.mutate();
    }
};
enum{sine = 0, sigmoid, gaussian, linear};

struct default_params{
    static constexpr float _rate_add_neuron = 0.1;
    static constexpr float _rate_del_neuron = 0.01;
    static constexpr float _rate_add_conn = 0.1;
    static constexpr float _rate_del_conn = 0.01;
    static constexpr float _rate_change_conn = 0.1;

    static constexpr size_t _min_nb_neurons = 2;
    static constexpr size_t _max_nb_neurons = 30;
    static constexpr size_t _min_nb_conns = 1;
    static constexpr size_t _max_nb_conns = 100;
};

}//cppn

// Activation function for Compositional Pattern Producing Network

struct AfCppn : public Af<cppn::Params> {

    float operator() (float p) const {
        float s = p > 0 ? 1 : -1;
        //std::cout<<"type:"<<this->_params.type()<<" p:"<<p<<" this:"<<this
        //<< " out:"<< p * exp(-powf(p * 10/ this->_params.param(), 2))<<std::endl;
        switch (this->_params.type) {
        case cppn::sine:
            return sin(_params.p0.data(0)*p + _params.p1.data(0));
        case cppn::sigmoid:
            return ((1.0 / (_params.p0.data(0) + exp(-p))) - _params.p1.data(0)) * 2.0;
        case cppn::gaussian:
            return exp(-_params.p0.data(0)*powf(p, 2)+_params.p1.data(0));
        case cppn::linear:
            return std::min(std::max(_params.p0.data(0)*p+_params.p1.data(0), -3.0f), 3.0f) / 3.0f;
        default:
            assert(0);
        }
        return 0;
    }
};



template <typename Params>
class CPPN : public NN<Neuron<PfWSum<EvoFloat<1,evo_float::default_params>>,AfCppn>, Connection<EvoFloat<1,evo_float::default_params>>>
{
public:

    CPPN(){}
    CPPN(size_t nb_inputs, size_t nb_outputs) :
        _nb_inputs(nb_inputs), _nb_outputs(nb_outputs)
    {   set_nb_inputs(_nb_inputs);
        set_nb_outputs(_nb_outputs);}

    void _random_neuron_params() {
      BGL_FORALL_VERTICES_T(v, this->_g, graph_t) {
        this->_g[v].get_pfparams().random();
        this->_g[v].get_afparams().random();
      }
    }

    void random(){
        // io
        this->set_nb_inputs(_nb_inputs);
        this->set_nb_outputs(_nb_outputs);
        _random_neuron_params();

        // neurons
        size_t nb_neurons = std::uniform_int_distribution<>(Params::_min_nb_neurons, Params::_max_nb_neurons)(rgen_t::gen);
        for (size_t i = 0; i < nb_neurons; ++i)
          _add_neuron();//also call the random params

        // conns
        size_t nb_conns = std::uniform_int_distribution<>(Params::_min_nb_conns, Params::_max_nb_conns)(rgen_t::gen);
        for (size_t i = 0; i < nb_conns; ++i)
          _add_conn();

        this->simplify();
    }

    void mutate(){
        _change_connections();

        _change_neurons();

        std::uniform_real_distribution<> dist(0,1);

        if (dist(rgen_t::gen) < Params::_rate_add_conn)
            _add_conn();

        if (dist(rgen_t::gen) < Params::_rate_del_conn)
            _del_conn();

        if (dist(rgen_t::gen) < Params::_rate_add_neuron)
            _add_neuron_on_conn();

        if (dist(rgen_t::gen) < Params::_rate_del_neuron)
            _del_neuron();

    }

    void _add_neuron(){
        vertex_desc_t n = add_neuron("n");
        this->_g[n].get_pfparams().random();
        this->_g[n].get_afparams().random();
    }

    void _add_neuron_on_conn(){
        if (!num_edges(this->_g))
          return;
        edge_desc_t e = random_edge(this->_g);
        vertex_desc_t src = source(e, this->_g);
        vertex_desc_t tgt = target(e, this->_g);
        weight_t w = this->_g[e].get_weight();
        vertex_desc_t n = add_neuron("n");
        this->_g[n].get_pfparams().random();
        this->_g[n].get_afparams().random();
        //
        remove_edge(e, this->_g);
        this->add_connection(src, n, w);// todo : find a kind of 1 ??
        this->add_connection(n, tgt, w);
    }

    void _del_neuron(){
        assert(num_vertices(this->_g));

        if (this->get_nb_neurons() <= this->get_nb_inputs() + this->get_nb_outputs())
            return;
        vertex_desc_t v;
        do
            v = random_vertex(this->_g);
        while (this->is_output(v) || this->is_input(v));

        clear_vertex(v, this->_g);
        remove_vertex(v, this->_g);
    }



    void _add_conn(){
        vertex_desc_t src, tgt;
        // this is only an upper bound; a connection might of course
        // be possible even after max_tries tries.
        size_t max_tries = num_vertices(this->_g) * num_vertices(this->_g),
                nb_tries = 0;
        do {
            src = _random_src();
            tgt = _random_tgt();
        } while (is_adjacent(_g, src, tgt) && ++nb_tries < max_tries);
        if (nb_tries < max_tries) {
            weight_t w;
            w.random();
            this->add_connection(src, tgt, w);
        }
    }

    void _del_conn(){
        if (!this->get_nb_connections())
            return;
        remove_edge(random_edge(this->_g), this->_g);
    }

    void _change_neurons(){
        BGL_FORALL_VERTICES_T(v, this->_g, graph_t) {
            this->_g[v].get_afparams().mutate();
            this->_g[v].get_pfparams().mutate();
        }
    }

    void _change_connections(){
        BGL_FORALL_EDGES_T(e, this->_g, graph_t)
                this->_g[e].get_weight().mutate();

        BGL_FORALL_EDGES_T(e, this->_g, graph_t)
                if (std::uniform_real_distribution<>(0,1)(rgen_t::gen) < Params::_rate_change_conn) {
                    vertex_desc_t src = source(e, this->_g);
                    vertex_desc_t tgt = target(e, this->_g);
                    weight_t w = this->_g[e].get_weight();
                    remove_edge(e, this->_g);
                    int max_tries = num_vertices(this->_g) * num_vertices(this->_g),
                    nb_tries = 0;
                    if (std::uniform_int_distribution<>(0,1)(rgen_t::gen))
                        do
                            src = _random_src();
                        while(++nb_tries < max_tries && is_adjacent(this->_g, src, tgt));
                    else
                        do
                            tgt = _random_tgt();
                        while(++nb_tries < max_tries && is_adjacent(this->_g, src, tgt));
                    if (nb_tries < max_tries)
                        this->add_connection(src, tgt, w);
                    return;
                }
    }



private:
    size_t _nb_inputs = 2;
    size_t _nb_outputs = 1;

    template <class Graph>
    typename boost::graph_traits<Graph>::vertex_descriptor
    random_vertex(Graph& g) {
        assert(num_vertices(g));
        int nbv = num_vertices(g);
        using namespace boost;
        if (num_vertices(g) > 1) {
            std::size_t n = std::uniform_int_distribution<>(0,num_vertices(g)-1)(rgen_t::gen);
            typename graph_traits<Graph>::vertex_iterator i = vertices(g).first;
            while (n-- > 0) ++i;
            return *i;
        } else
            return *vertices(g).first;
    }

    template <class Graph>
    typename boost::graph_traits<Graph>::edge_descriptor
    random_edge(Graph& g) {
        assert(num_edges(g));
        using namespace boost;
        if (num_edges(g) > 1) {
            std::size_t n = std::uniform_int_distribution<>(0,num_edges(g)-1)(rgen_t::gen);
            typename graph_traits<Graph>::edge_iterator i = edges(g).first;
            while (n-- > 0) ++i;
            return *i;
        } else
            return *edges(g).first;
    }

    vertex_desc_t _random_src(){
        vertex_desc_t v;
        do
            v = random_vertex(this->_g);
        while (this->is_output(v));
        return v;
    }

    vertex_desc_t _random_tgt(){
        vertex_desc_t v;
        do
            v = random_vertex(this->_g);
        while (this->is_input(v));
        return v;
    }
};


}//nn2

#endif //CPPN_HPP_
