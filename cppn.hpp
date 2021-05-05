#ifndef CPPN_HPP_
#define CPPN_HPP_

#include "nn.hpp"
#include "evo_float.hpp"
#include "random.hpp"
#include <random>

using namespace boost;

namespace nn2{

namespace cppn{

typedef nn2::EvoFloat<1,evo_float::default_params> evo_float_t;

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
class CPPN : public NN<Neuron<PfWSum<cppn::evo_float_t>,AfCppn>, Connection<cppn::evo_float_t>>
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

        // serialize the graph "by hand"...
              template<typename Archive>
              void save(Archive& a, const unsigned v) const {
    //            dbg::trace("cppn", DBG_HERE);
                std::vector<int> inputs;
                std::vector<int> outputs;
                std::vector<typename neuron_t::af_t::params_t> afparams;
                std::vector<typename neuron_t::pf_t::params_t> pfparams;
                std::map<vertex_desc_t, int> nmap;
                std::vector<std::pair<int, int> > conns;
                std::vector<weight_t> weights;

                BGL_FORALL_VERTICES_T(v, this->_g, graph_t) {
                  if (this->is_input(v))
                    inputs.push_back(afparams.size());
                  if (this->is_output(v))
                    outputs.push_back(afparams.size());
                  nmap[v] = afparams.size();
                  afparams.push_back(this->_g[v].get_afparams());
                  pfparams.push_back(this->_g[v].get_pfparams());
                }
                BGL_FORALL_EDGES_T(e, this->_g, graph_t) {
                  conns.push_back(std::make_pair(nmap[source(e, this->_g)],
                                                 nmap[target(e, this->_g)]));
                  weights.push_back(this->_g[e].get_weight());
                }
                assert(pfparams.size() == afparams.size());
                assert(weights.size() == conns.size());

                a & BOOST_SERIALIZATION_NVP(afparams);
                a & BOOST_SERIALIZATION_NVP(pfparams);
                a & BOOST_SERIALIZATION_NVP(weights);
                a & BOOST_SERIALIZATION_NVP(conns);
                a & BOOST_SERIALIZATION_NVP(inputs);
                a & BOOST_SERIALIZATION_NVP(outputs);
              }
              template<typename Archive>
              void load(Archive& a, const unsigned v) {
//                dbg::trace("nn", DBG_HERE);
                std::vector<int> inputs;
                std::vector<int> outputs;
                std::vector<typename neuron_t::af_t::params_t> afparams;
                std::vector<typename neuron_t::pf_t::params_t> pfparams;
                std::map<size_t, vertex_desc_t> nmap;
                std::vector<std::pair<int, int> > conns;
                std::vector<weight_t> weights;

                a & BOOST_SERIALIZATION_NVP(afparams);
                a & BOOST_SERIALIZATION_NVP(pfparams);
                a & BOOST_SERIALIZATION_NVP(weights);
                a & BOOST_SERIALIZATION_NVP(conns);
                a & BOOST_SERIALIZATION_NVP(inputs);
                a & BOOST_SERIALIZATION_NVP(outputs);

                assert(pfparams.size() == afparams.size());

                assert(weights.size() == conns.size());
                this->set_nb_inputs(inputs.size());
                this->set_nb_outputs(outputs.size());
                for (size_t i = 0; i < this->get_nb_inputs(); ++i)
                  nmap[inputs[i]] = this->get_input(i);
                for (size_t i = 0; i < this->get_nb_outputs(); ++i)
                  nmap[outputs[i]] = this->get_output(i);

                for (size_t i = 0; i < afparams.size(); ++i)
                  if (std::find(inputs.begin(), inputs.end(), i) == inputs.end()
                      && std::find(outputs.begin(), outputs.end(), i) == outputs.end())
                    nmap[i] = this->add_neuron("n", pfparams[i], afparams[i]);
                  else {
                    this->_g[nmap[i]].set_pfparams(pfparams[i]);
                    this->_g[nmap[i]].set_afparams(afparams[i]);
                  }


                //assert(nmap.size() == num_vertices(this->_g));
                for (size_t i = 0; i < conns.size(); ++i)
                  this->add_connection(nmap[conns[i].first], nmap[conns[i].second], weights[i]);
              }
              BOOST_SERIALIZATION_SPLIT_MEMBER();

private:

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
public:
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
        size_t max_tries = num_vertices(this->_g) * num_vertices(this->_g),
                nb_tries = 0;
        do {
            src = _random_src();
            tgt = _random_tgt();
        } while (src == tgt && ++nb_tries < max_tries);
        weight_t w;
        w.random();
        this->add_connection(src, tgt, w);
    }

    void _add_conn_nodup(){
        vertex_desc_t src, tgt;
        // this is only an upper bound; a connection might of course
        // be possible even after max_tries tries.
        size_t max_tries = num_vertices(this->_g) * num_vertices(this->_g),
                nb_tries = 0;
        do {
            src = _random_src();
            tgt = _random_tgt();
        } while (src == tgt && is_adjacent(_g, src, tgt) && ++nb_tries < max_tries);
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


};


}//nn2

#endif //CPPN_HPP_
