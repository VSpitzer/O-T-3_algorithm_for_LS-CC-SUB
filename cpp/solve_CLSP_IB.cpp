#include <vector>
#include <valarray> 
#include <algorithm>  
#include <math.h>                
#include <limits>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <random>

#include <chrono>

#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/labeled_graph.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>

#include <cmath>

#include <map>
#include <tuple>


#include <omp.h>

using namespace std;
using namespace boost;
using namespace std::chrono;

#define NUM_THREADS 1

// Define the graph type with edge weights
typedef property<vertex_distance_t, float> VertexProperty;



typedef adjacency_list<vecS, vecS, directedS,
                       no_property,
                       property<edge_weight_t, float>> Graph;


// Define the graph type using compressed_sparse_row_graph
typedef boost::compressed_sparse_row_graph<boost::directedS, // Directed graph
                                           boost::no_property, // No properties for vertices
                                           boost::property<boost::edge_weight_t, float> // Edge weight property
                                           > CompressedGraph;

// Define types for vertices and edges
typedef boost::graph_traits<CompressedGraph>::vertex_descriptor Vertex;
typedef boost::graph_traits<CompressedGraph>::edge_descriptor Edge;

typedef vector<float> vec;
typedef vector<int> ivec;
typedef map<int, float> dict;

// Helper function to compute the sum of a vector
float sum(const vector<int>& v) {
    float total = 0;
    for (auto& el : v) total += el;
    return total;
}

// Helper function to find minimum index in vector
int argmin(const vector<float>& v) {
    int t=0;
    float min_val = std::numeric_limits<float>::infinity() ;
    int min_id = 0 ;
    for (float ev : v){
        if (ev<min_val){
            min_val = ev;
            min_id = t;
        }
        t+=1;
    }
    return min_id;
}

int argmax(const vector<float>& v) {
    int t=0;
    float max_value = - std::numeric_limits<float>::infinity() ;
    int max_id = 0 ;
    for (float ev : v){
        if (ev>max_value){
            max_value = ev;
            max_id = t;
        }
        t+=1;
    }
    return max_id;
}

int min_val(int a, int b) {
    int res = a;
    if (b<a){
        res = b;
    }
    return res;
}

int max_val(int a, int b) {
    int res = a;
    if (b>a){
        res = b;
    }
    return res;
}

float abs_val(float a) {
    float res = a;
    if (res<0){
        res = - a;
    }
    return res;
}

void normalize_ini_stock(vector<int>& d, vector<int>& S, int T, int s_i){

    vector<int> sum_d(T+1,0);
    for (int t=0; t<T; t++){
        sum_d[t+1] = sum_d[t] + d[t];
        d[t] = max_val(0, d[t]-max_val(0,s_i-sum_d[t]));
        S[t] = S[t]-max_val(0,s_i-sum_d[t+1]);
        if(S[t]<0){
            printf("Unfeasible problem: initial storage value \n");
        }
    }

    return ;
}

void normalize_demand(vector<int>& d, vector<int>& S, int T, int C){
    
    vector<int> I_bar(T,0);
    int t=0;
    for (int t2=0; t2<T-1; t2++){
        t=T-2-t2;
        I_bar[t] = -min_val(0, -(I_bar[t+1]+d[t+1]-C));
    }
    for (t=0; t<T; t++){
        d[t] = min_val(C, d[t]+I_bar[t]);
        S[t] = S[t]-I_bar[t];
    }
    

    return ;
}

void normalize_bounds(vector<int>& d, vector<int>& S, int T, int C ){
    
    S[0] = min_val(S[0], C-d[0]);
    for (int t=1; t<T; t++){
        S[t] = min_val(S[t], S[t-1]+C-d[t]);
    }
    for (int t=T-2; t>=0; t-=1){
        S[t] = min_val(S[t], S[t+1]+d[t+1]);
    }

    return ;
}

void solve_T2(vector<int>& res, int T, int I1, int s_out, vector<int> S, vector<int> d, vector<int> sum_d, int total_d, int K, int f, int C, int x_on, vector<float> cf, int t0) {

    if (K + (int)(f > 0) > T || total_d < 0) {
        for (int t=0; t<T; ++t){
            res[t]=1;
        }
        return;  // Return vector of ones
    } else if (K == 0 && f == 0) {
        return;  // Return vector of zeros
    } 

    float min_cost = cf[argmin(cf)];
    float min_value = abs_val((f + C + x_on)*min_cost)+1 ;

    vector<pair<int, int>> edges;
    vector<float> weights;

    int num_vertices = 2 * T * (K + 1) + 1;      
    int init_state = 2 * T * (K + 1);
    if (I1 - d[0] >= 0 && I1 - d[0] <= S[0]) {
        edges.push_back({init_state, 0});
        weights.push_back(min_value);
    }
    if (K>0 && I1 + C - d[0] >= 0 && I1 + C - d[0] <= S[0]) {
        edges.push_back({init_state, 1});
        weights.push_back((C + x_on)*cf[0]+min_value);
    }
    if (I1 + f - d[0] >= 0 && I1 + f - d[0] <= S[0] &&  (0 == t0 || f == C)) {
        edges.push_back({init_state, (K+1) });
        weights.push_back(min_value);
    }
    if (K>0 && I1 + C + f - d[0] >= 0 && I1 + C + f - d[0] <= S[0] && 0 == t0) {
        edges.push_back({init_state, (K+1) +1});
        weights.push_back((C + x_on)*cf[0]+min_value);
    }
    int s_t = 0;
    int dk = 0;
    for (int t = 0; t < T-1; t++) {  
        for (int k = 0; k <= K; k++) {
            for (int i = 0; i <= 1; i++) {
                // Compute s_t
                s_t = I1 + k * C + i * f - sum_d[t+1] - s_out*int(t==T-1)*S[T-1];
                // No change
                if (s_t - d[t+1] >= 0 && s_t - d[t+1] <= S[t+1]) {
                    edges.push_back({t * 2 * (K+1) + i * (K+1)  + k, (t + 1) * 2 * (K+1)  + i * (K+1)  + k});
                    weights.push_back(min_value);
                }
                // One fractional production
                if (s_t + f - d[t+1] >= 0 && s_t + f - d[t+1] <= S[t+1] && i == 0 && (t+1 == t0 || f == C)) {
                    edges.push_back({t * 2 * (K+1)  + k, (t + 1) * 2 * (K+1)  + (K+1)  + k});
                    weights.push_back(min_value);
                }
                // One full production
                if (K>0 && k<K && s_t + C - d[t+1] >= 0 && s_t + C - d[t+1] <= S[t+1]) {
                    edges.push_back({t * 2 * (K+1)  + i * (K+1)  + k, (t + 1) * 2 * (K+1)  + i * (K+1)  + (k + 1)});
                    weights.push_back((C + x_on)*cf[t+1]+min_value);
                }
                // Both full and fractional production
                if (K>0 && f < C && k<K && s_t + f + C - d[t+1] >= 0 && s_t + f + C - d[t+1] <= S[t+1] && i == 0 && t+1 == t0) {
                    edges.push_back({t * 2 * (K+1)  + k, (t + 1) * 2 * (K+1)  + (K+1)  + (k + 1)});
                    weights.push_back((C + x_on)*cf[t+1]+min_value);
                }
            }
        }
    }
    CompressedGraph G(boost::edges_are_unsorted_multi_pass, edges.begin(), edges.end(),
        weights.begin(), num_vertices);


    // Distance vector (initialized to infinity)
    std::vector<float> distances(num_vertices, std::numeric_limits<float>::infinity());

    // Predecessor vector (initialized to null vertex)
    std::vector<Vertex> predecessors(num_vertices, boost::graph_traits<CompressedGraph>::null_vertex());

    distances[init_state] = 0.0f; // Distance from source to source is 0
    assert(init_state == num_vertices - 1);
    bool success = bellman_ford_shortest_paths(G, weight_map(get(edge_weight, G))
        .predecessor_map(&predecessors[0])
        .distance_map(&distances[0]));

    if (!success) {
        throw runtime_error("Negative cycle detected");
    }

    int target_vertex = -1;
    if(f>0){
        target_vertex = (T-1) * 2 * (K+1)  + (K+1)  + K ; 
    }else{
        target_vertex = (T-1) * 2 * (K+1)  + K ; 
    }
    std::vector<int> path;
    int t = T-1;
    int t_f = -1;
    int state = target_vertex;
    for (int v = target_vertex; v > 2*(K+1) ; v = predecessors[v]) { // Start from the target and go backwards
        path.push_back(v); // Add current vertex to the path
        if ((K>0) && (((v - predecessors[v] - 2 * (K+1)  == 1) || (v - predecessors[v] - 2 * (K+1)  == (K+1)  && f == C) ||  (v  - predecessors[v] - 2 * (K+1)  == 1 + (K+1) )))) {
            res[t] = 1;
        }
        if ((v - predecessors[v] - 2 * (K+1)  == (K+1) ) ||  (v  - predecessors[v] - 2 * (K+1)  == 1 + (K+1)  && K > 0)) {
            t_f=t;
        }
        t -= 1;
        state = predecessors[v];
    }
    if ( K>0 && (((state == 1) || (state == (K+1)  + 1)) )){
        res[0] = 1;
    }
    if ((state== (K+1) ) ||  (state == 1 + (K+1)  && K > 0)) {
        t_f=0;
    }

    path.push_back(0); // Add the source vertex (0) at the end
    std::reverse(path.begin(), path.end()); // Reverse the path to get the correct order

    return;
}

int init_solve_T3(vector<int>& start, vector<vector<int>>& res, int T_all, vector<int> S_all, vector<int> d_all, int C, int x_on, vector<float> cf, int t1, int s_in, int s_out) {

    int t2 = T_all;

    int I1 = 0;
    int It1 = 0;
    if (t1 > 0) {
        I1 = s_in * S_all[t1 - 1];
        It1 = S_all[t1 - 1];
    } else {
        I1 = 0;
        It1 = 0;
    }
    int I2 = s_out * S_all[t2 - 1];

    int T = t2 - t1;
    vector<int> d(d_all.begin() + t1, d_all.end());
    vector<int> S(S_all.begin() + t1, S_all.end());
    
    vector<int> d_p(T);
    int d_p_1 = d[0] + S[0];
    if (t1 > 0) {
        d_p_1 = d[0] - S_all[t1 - 1] + S[0];
    }
    d_p[0] = d_p_1;
    for (int t = 1; t < T; ++t) {
        d_p[t] = d[t] - S[t - 1] + S[t];
    }

    vector<int> sum_d = vector<int>(T+1, 0);
    for (int v = 1; v < T+1; v += 1) { 
        sum_d[v]=sum_d[v-1]+d[v-1];
    }

    vector<int> delta_d(T,0);
    for (int t = 0; t < T; ++t) {
        if (sum_d[t+1] - I1>0){
           delta_d[t] = sum_d[t+1] - I1;
        }
    }
    delta_d[T - 1] = sum_d[T] + I2 - I1;

    int total_d = delta_d[T - 1] ;
    int K = (int)(total_d*1./C);  
    int f = total_d-K*C;  

    //Manage special cases
    if (K + (int)(f > 0) > T || total_d < 0) {
        return T_all+1;  // Return infeasability
    } 

    int t0 = 0;
    for (int t = 1; t < T; ++t) {
        if (I2 + sum_d[T] - sum_d[t] >= f) {
            t0 = t;
        }
    }

    // Call solve_T2
    vector<int> P_prime(T, 0);
    solve_T2(P_prime, T, I1, s_out, S, d, sum_d, total_d, K, f, C, x_on, cf, t0);

    copy(P_prime.begin(), P_prime.end(), res[T-1].begin());

    start[T-1] = t0;

    // Dynamic Initialization
    vector<int> I(T, 0);
    vector<int> J(T + 1, 0);

    vector<int> t2_list;
    for (int t2i = t1+1; t2i < T_all; ++t2i) {
        t2_list.push_back(t1 + T_all - t2i);
    }

    int t_J = t0;
    int t2_min = T_all;

    int inventory = 0;

    int K_new = 0;
    int f_new = 0;
    bool change = false;
    int t_change = 0;
    for (int t2 : t2_list) {

        T = t2 - t1;
        I2 = s_out * S_all[t2 - 1];
        delta_d[T - 1] = sum_d[T] + I2 - I1;
        total_d = delta_d[T - 1];

        K_new = (int)(total_d*1./C);  
        f_new = total_d-K_new*C; 

        I[0] = I1 + P_prime[0] * C + f_new * (t0 == 0) - d[0];
        J[T - 1] = (S_all[t2 - 1] - I2) + P_prime[T - 1] * C + f_new * (t0 >= T-1) - d_p[T - 1];
        for (int t = 1; t < T; ++t) {
            I[t] = I[t - 1] + P_prime[t] * C + f_new * (t0 == t) - d[t];
            J[T - t - 1] = J[T - t] + P_prime[T - t - 1] * C + f_new * (t0 == T - t - 1) - d_p[T - t - 1];
        }


        change = false;
        t_change = -1;

        while (I2 + sum_d[T] - sum_d[t0] < f_new || t0 > T-1) {
            if (P_prime[t0] == 1) {
                change = true;
                t_change = t0;
            }
            t0 -= 1;
            I[t0] += f_new;
            J[t0+1] -= f_new;
        }

        if (t0 == -1) {
            break;
        }

        t_J = t0;
        int t_opt = -1;

        if (change && K_new == K) {
            float min_cost = numeric_limits<float>::infinity();
            while (t_J >= 0 && (J[t_J+1] >= 0 || t_J == T-1)) {
                if (P_prime[t_J] == 0 && cf[t_J]  + cf[t_J]*x_on*1./C < min_cost) {
                    t_opt = t_J;
                    min_cost = cf[t_J]  + cf[t_J]*x_on*1./C ;
                }
                t_J -= 1;
            }
            if (t_opt == -1) {
                break;
            }
            P_prime[t_opt] = 1;
            P_prime[t_change]=0;
        } else if (change && K_new < K) {
            P_prime[t_change]=0;
        } else if (!change && K_new < K) {
            float max_cost = -numeric_limits<float>::infinity();
            while (t_J >= 0 && (I[t_J] - C >= 0 || t_J == T-1)) {
                if (P_prime[t_J] == 1 && cf[t_J]  + cf[t_J]*x_on*1./C  > max_cost) {
                    t_opt = t_J;
                    max_cost = cf[t_J]  + cf[t_J]*x_on*1./C;
                }
                t_J -= 1;
            }
            if (t_opt == -1) {
                break;
            }
            P_prime[t_opt] = 0;
        }

        copy(P_prime.begin(), P_prime.end(), res[T-1].begin());
        start[T-1] = t0;

        K = K_new;
        f = f_new;

        t2_min = t2;


    }

    return t2_min ;
}

float dyna_solve_T3(vector<int>& sol, vector<int> S_all, vector<int> d_all, int C, int x_on, float max_value, vector<float> cf, int t1, int t2, int s_in, int s_out, int t0, vector<int> P_prime) {

    int I1 = (t1 > 0) ? s_in * S_all[t1 - 1] : 0;
    int It1 = (t1 > 0) ? S_all[t1 - 1] : 0;
    int I2 = s_out * S_all[t2 - 1];
    int It2 = S_all[t2 - 1];

    int T = t2 - t1;
    vector<int> d(d_all.begin() + t1, d_all.begin() + t2 );
    vector<int> S(S_all.begin() + t1, S_all.begin() + t2 );

    vector<int> sum_d = vector<int>(T, 0);
    sum_d[0] = d[0];
    for (int v = 1; v < T; v += 1) { 
        sum_d[v]=sum_d[v-1]+d[v];
    }

    vector<int> delta_d(T,0);
    for (int t = 0; t < T; ++t) {
        if (sum_d[t] - I1>0){
           delta_d[t] = sum_d[t] - I1;
        }
    }
    delta_d[T - 1] = sum_d[T-1] + I2 - I1;

    int total_d = delta_d[T - 1] ;
    int K = (int)(total_d*1./C);  
    int f = total_d-K*C; 

    int t_I_min = T;
    for (int t = 0; t < T; ++t) {
        if (t*C+f+I1>=sum_d[t]){
            t_I_min = t;
            break;
        }
    }
    int t_J_min=T;
    for (int t = 0; t < T; ++t) {
        if (I1+f-sum_d[t]<=S[t]){
            t_J_min = t;
            break;
        }
    }
    int t_min = max(t_I_min,t_J_min);


    auto SD = [&](int i) { return i < 0 ? 0 : sum_d[i]; };
    auto SS = [&](int i) { return i < 0 ? It1 : S[i];     };
    int t_I_max = -1;
    for (int t = T - 1; t >= 0; t -= 1) {
        if (I2 + sum_d[T - 1] - SD(t - 1) >= f) {
            t_I_max = t;
            break;
        }
    }
    int t_J_max = -1;
    for (int t = T - 1; t >= 0; t -= 1) {
        if (I1 + (K - (T - 1 - t)) * C - SD(t - 1) <= SS(t - 1)) {
            t_J_max = t;
            break;
        }
    }
    int t_max = min_val(t_J_max,t_I_max);
        
    // Handle special cases
    if (K == 0 && f == 0) {
        fill(sol.begin(), sol.end(), 0);
        float sol_cost = 0.;
        for (int t=0; t<T; t++) {
            if (cf[t]<0)
            {
                sol_cost += cf[t]*x_on;
            }
        }
        return sol_cost;
    } else if (K > 0 && f == 0) {
        transform(P_prime.begin(), P_prime.end(), sol.begin(), [&](int p) { return p * C; });
        float sol_cost = 0.;
        for (int t=0; t<T; t++){
            sol_cost += (C+x_on)*cf[t]*P_prime[t];
            if (cf[t]<0 && P_prime[t]==0)
            {
                sol_cost += cf[t]*x_on;
            }
        }
        return sol_cost;
    } else if (K == 0 && f > 0) {
        int t_opt = t_J_min + argmin(vector<float>(cf.begin()+t_J_min,cf.begin()+t0+1));
        sol[t_opt] = f;
        float cost = (f + x_on) * cf[t_opt];
        for (int t=0; t<T; t++) {
            if (cf[t]<0 && sol[t]==0)
            {
                cost += cf[t]*x_on;
            }
        }
        return cost;
    }else if (K + int(f > 0) > T || total_d < 0 || t_max==-1) {
        fill(sol.begin(), sol.end(), C);
        return max_value;
    }

    //Evaluate initial P_prime cost
    float cost_P_prime = 0.0;
    for (int t=0; t<T; t++){
        cost_P_prime += (C+x_on)*cf[t]*P_prime[t];
    }


    // Determine I and J
    vector<int> I(T, 0);
    vector<int> J(T, 0);
    I[0] = I1 + P_prime[0] * C + f * (t0 == 0) - d[0];
    J[0] = It1 - I1;
    for (int t = 1; t < T; ++t) {
        I[t] = I[t - 1] + P_prime[t] * C + f * (t0 == t) - d[t];
        J[t] = S[t-1]-I[t-1];
    }



    // Compute s, determine gamma
    int s = 0;
    for (int t = 0; t <= t0 - 1; ++t) {
        int t_r = t0 - t;
        if (I[t_r - 1] < C) {
            s = t_r;
            break;
        }
    }

    vector<int> gamma(T, -1);
    if (s <= t0) {
        gamma[s] = s * (P_prime[s] == 1) - 1*(P_prime[s] == 0);
        for (int t = s + 1; t <= t0; ++t) {
            if (P_prime[t] == 1) {
                if (gamma[t - 1] > -1) {
                    if (cf[gamma[t - 1]]*(C+x_on) > cf[t]*(C+x_on)) {
                        gamma[t] = gamma[t - 1];
                    }else{
                        gamma[t] = t;
                    }
                } else {
                    gamma[t] = t;
                }
            } else {
                gamma[t] = gamma[t - 1];
            }
        }
    }

    // Compute u, determine delta
    int u = T-1;
    for (int t = t0; t < T; ++t) {
        if (I[t] < C) {
            u = t;
            break;
        }
    }
    int delta = -1;
    for (int t = t0+1; t <= u; ++t) {
        if (P_prime[t] == 0) {
            if (delta == -1) {
                delta = t;
            }else if ( cf[t] < cf[delta]){
                delta = t;
            }
        }
    }

    // Compute q, determine eta
    int q = 0;
    for (int t = 1; t <= t0; ++t) {
        if (J[t] < C) {
            q = t;
        }
    }

    vector<vector<int>> eta(T, vector<int>(T, -1));
    eta[q][q] = q * (P_prime[q] == 0) - 1*(P_prime[q] == 1) ;

    for (int t = q + 1; t <= t0; ++t) {
        if (P_prime[t] == 0) {
            if (eta[q][t - 1]>-1 && cf[eta[q][t - 1]]*(C+x_on) < cf[t]*(C+x_on)) {
                eta[q][t] = eta[q][t - 1];
            }else{
                eta[q][t] = t;
            }
        } else {
            eta[q][t] = eta[q][t - 1];
        }
    }

    // Find optimal solution for initial iteration
    float min_cost = std::numeric_limits<float>::infinity();
    int t_opt = t0;
    int change_P_opt = -1;
    float P_cost = cost_P_prime + (f + x_on) * cf[t0];
    int change_P = -1;

    if (P_prime[t0] > 0) {
        P_cost = P_cost - (C + x_on) * cf[t0];
        if (eta[q][t0] > -1 && delta == -1) {
            P_cost = P_cost + (C + x_on) * cf[eta[q][t0]];
            change_P = eta[q][t0];
        } else if (eta[q][t0] == -1 && delta > -1 ){
            P_cost = P_cost + (C + x_on) * cf[delta];
            change_P = delta;
        } else if (eta[q][t0] > -1 && delta > -1 && cf[eta[q][t0]]<cf[delta]){
            P_cost = P_cost + (C + x_on) * cf[eta[q][t0]];
            change_P = eta[q][t0];
        } else if (eta[q][t0] > -1 && delta > -1 && cf[eta[q][t0]]>cf[delta]){
            P_cost = P_cost + (C + x_on) * cf[delta];
            change_P = delta;
        } else{
            P_cost = max_value * T;
            if(t0<=t_max){
                printf("Error init unfeasible P \n");
            }
        }
    }


    if (min_cost > P_cost) {
        min_cost = P_cost;
        t_opt = t0;
        change_P_opt = change_P;
    }
        
    //defined move
    bool ready_opt_move = true;
	int gamma_m = -1;
    bool opt_move = true;
    bool feasibility_move = true;
    vector<std::pair<int, int>> change = { {-1, -1} };
    int q_new = q;

    //Loop
    while (t0 > t_min) {

        // Update t0
        t0--;

        // Update delta
        if (P_prime[t0 + 1] == 0) {
            if (delta > -1) {
                if (cf[delta]*(C+x_on) > cf[t0 + 1]*(C+x_on)) {
                    delta = t0 + 1;
                }
            } else {
                delta = t0 + 1;
            }
        }

        // Update I and J
        I[t0] += f;
        J[t0 + 1] -= f;

        // Update u
        if (I[t0] < C-1e-2) {
            u = t0;
            delta = t0 * (P_prime[t0] == 0) - 1*(P_prime[t0] == 1);
        }

        // Update s and gamma
        if (t0 < s) {

            s = 0;
            for (int t = 0; t <= t0 - 1; ++t) {
                int t_r = min(T-1,t0 - t);
                if (I[t_r - 1] < C-1e-2) {
                    s = t_r;
                    break;
                }
            }

            fill(gamma.begin(), gamma.end(), -1);
            if (s <= t0) {
                gamma[s] = s * (P_prime[s] ==1) - 1*(P_prime[s] ==0);
                for (int t = s + 1; t <= t0; ++t) {
                    if (P_prime[t] == 1) {
                        if (gamma[t - 1] > -1) {
                            if (cf[gamma[t - 1]]*(C+x_on) > cf[t]*(C+x_on)){
                                gamma[t] = gamma[t - 1];
                            }else{
                                gamma[t] = t;
                            }
                        } else {
                            gamma[t] = t;
                        }
                    } else {
                        gamma[t] = gamma[t - 1];
                    }
                }
            }
        }

        if (!ready_opt_move && t0 < gamma_m) {
            ready_opt_move = true;
        }

        // Update q and eta
        if (q == t0 + 1) {

            q = 0;
            for (int t = 0; t <= t0; ++t) {
                if (J[t0 - t] < C-1e-2) {
                    q = t0 - t;
                    break;
                }
            }
            eta[q][q] = q * (P_prime[q] == 0)  - 1*(P_prime[q] == 1);
            for (int t = q + 1; t <= t0 + 1; ++t) {
                if (P_prime[t] == 0) {
                    if (eta[q][t - 1]>-1 && cf[eta[q][t - 1]]*(C+x_on) < cf[t]*(C+x_on)){
                        eta[q][t] = eta[q][t - 1];
                    }else{
                        eta[q][t] = t;
                    }
                } else {
                    eta[q][t] = eta[q][t - 1];
                }
            }
            
        }

        // Feasibility and Optimization Move
        feasibility_move = false;
        opt_move = false;
        if (ready_opt_move) {
            if (delta > -1 && gamma[t0] > -1 && cf[gamma[t0]]*(C+x_on) > cf[delta]*(C+x_on)) {
                P_prime[gamma[t0]] = 0;
                P_prime[delta] = 1;
                change.insert(change.begin(), {gamma[t0], delta});
                ready_opt_move = false;
				gamma_m = gamma[t0];
                opt_move = true;
            }
        }
        if (J[t0 + 1] < -1e-2 && !opt_move){
            if (delta>-1 && gamma[t0]>-1) {
                P_prime[gamma[t0]] = 0;
                P_prime[delta] = 1;
                change.insert(change.begin(), {gamma[t0], delta});
                feasibility_move = true;
            }
        }

        // Update data after move
        if (feasibility_move || opt_move) {
            for (int t = gamma[t0]; t <= t0; ++t) {
                I[t] -= C;
                if (t < T) {
                    J[t + 1] += C;
                }
            }
            cost_P_prime = cost_P_prime - (C + x_on) * cf[change[0].first] + (C + x_on) * cf[change[0].second];

            if (gamma[t0] < q) q = gamma[t0];
            eta[q][q] = q * (P_prime[q] == 0)  - 1*(P_prime[q] == 1);
            for (int t = q + 1; t <= t0 ; ++t) {
                if (P_prime[t] == 0) {
                    if (eta[q][t - 1]>-1 && cf[eta[q][t - 1]]*(C+x_on)  < cf[t]*(C+x_on)){
                        eta[q][t] = eta[q][t - 1];
                    }else{
                        eta[q][t] = t;
                    }
                } else {
                    eta[q][t] = eta[q][t - 1];
                }
            }

            u = t0;


            delta = t0 * (P_prime[t0] == 0) - 1*(P_prime[t0] == 1);
        } else {
            change.insert(change.begin(), {-1, -1});
        }

        // Find optimal solution for current iteration
        P_cost = cost_P_prime + (f + x_on) * cf[t0];
        change_P = -1;  

        // Check if P_prime at t0 is greater than 0
        if (P_prime[t0] > 0) {
            // Update P_cost by subtracting the switching cost at t0
            P_cost = P_cost - (C + x_on) * cf[t0];

            if (t0==0){
                if (delta==-1) {
                    P_cost = max_value * T; 
                }else{
                    P_cost = P_cost + (C + x_on) *cf[delta];  
                    change_P = delta;
                }
            }else{
                int best_change =-1;
                if (eta[q][t0-1]>-1){
                    best_change =eta[q][t0-1];
                } 


                if (best_change == -1 && delta==-1) {
                    if(t0<=t_max){
                        printf("Error unfeasible P \n");
                    }
                    P_cost = max_value * T; 
                } else if (best_change== -1 && delta>-1){
                    P_cost = P_cost + (C + x_on) *cf[delta];  
                    change_P = delta;
                }else if (best_change > -1 && delta==-1){
                    P_cost = P_cost + (C + x_on) *cf[best_change];  
                    change_P = best_change;
                }else if (best_change> -1 && delta>-1 && cf[best_change] >= cf[delta] ){
                    P_cost = P_cost + (C + x_on) *cf[delta]; 
                    change_P = delta; 
                }else if (best_change> -1 && delta>-1 &&  cf[best_change] < cf[delta] ){
                    P_cost = P_cost + (C + x_on) *cf[best_change];  
                    change_P = best_change;
                }
            }
            
        }

        // Compare current P_cost with min_cost, and update if current iteration has a lower cost
        if (min_cost > P_cost) {
            min_cost = P_cost;
            t_opt = t0;              // Update t_opt to the current t0
            change_P_opt = change_P;  // Store the optimal change period
        }

    }

    // Iterate through the changes array from index 1 to t_opt - t_min
    int cpt = 1;
    for (const auto& change_pair : change) {
        if (cpt > t_opt-t_min){
            break;
        }
        int i = change_pair.first;
        int j = change_pair.second;
        if (i > -1 && j > -1) {
            P_prime[i] = 1;  // Set P_prime[i] to 1 
            P_prime[j] = 0;  // Set P_prime[j] to 0
        }
        cpt+=1;
    }


    // Create the solution array and scale it by C
    for (int t=0; t<T; t++) {
        sol[t] = P_prime[t]*C;
    }


    // Set sol[t_opt] to f
    sol[t_opt] = f;  // Adjust for 0-based index in C++

    // If change_P_opt is not -1 (indicating a valid value), set sol[change_P_opt] to C
    if (change_P_opt != -1) {
        sol[change_P_opt ] = C;  // Adjust index for 0-based C++
    }

    for (int t=0; t<T; t++) {
        if (cf[t]<0 && sol[t]==0)
        {
            min_cost += cf[t]*x_on;
        }
    }

    return min_cost;
}


void CLSP_T3(vector<int>& solution, int T, vector<float> cf, vector<int> S, vector<int> d, int C, int x_on) {

    omp_set_num_threads(NUM_THREADS);

    float max_cf = cf[argmax(cf)] ;
    float min_cf = cf[argmin(cf)] ;
    if (-min_cf>max_cf){
        max_cf=-min_cf;
    }
    float max_val = T*abs_val((C + x_on)*max_cf) ;

    Graph G(2 * (T+1));
    int init_state = 0 ;

    vector<vector<vector<int>>> sol(2 * (T + 1), vector<vector<int>>(2 * (T + 1), vector<int>(T, 0)));
    vector<vector<float>> tab_cost(2 * (T + 1), vector<float>(2 * (T + 1), max_val));

    int n_th = NUM_THREADS;
    vector<int> tab_t1(T,0);
    for (int t=0; t<T; ++t){
        tab_t1[t] = (t*NUM_THREADS)%T + int((t*NUM_THREADS)*1./T);
    }

    #pragma omp parallel default(none) shared(T, tab_t1, S, d, C, x_on, max_val, cf, G, sol, tab_cost)
    {
        #pragma omp for
        for (int t1_tmp=0; t1_tmp<=T-1; t1_tmp+=1){
            int t1 = tab_t1[t1_tmp];
            for (int i=0; i<=3; i+=1){
                std::vector<int> start(T, 0);
                std::vector<std::vector<int>> res(T, std::vector<int>(T, 0));
                int t2_min = init_solve_T3(start, res, T, S, d, C, x_on, vec(cf.begin()+t1, cf.end()), t1, i%2, (int)(i*0.5));
                for (int t2=t2_min; t2<=T; t2+=1){
                    vector<int> tmp_sol(t2-t1,0);
                    std::vector<int> P_prime(res[t2 - t1 - 1].begin(), res[t2 - t1 - 1].begin() + t2 - t1);
                    float cost = dyna_solve_T3(tmp_sol, S, d, C, x_on, max_val, vec(cf.begin()+t1, cf.end()), t1, t2, i%2, (int)(i*0.5), start[t2-t1-1], P_prime);
                    add_edge(2*t1+i%2, 2*t2+(int)(i*0.5), cost, G);
                    copy(tmp_sol.begin(), tmp_sol.end(), sol[2*t1+i%2][2*t2+(int)(i*0.5)].begin() + t1);
                    tab_cost[2*t1+i%2][2*t2+(int)(i*0.5)] = cost;

                }
            }
        }
    } 


    // Bellman-Ford Shortest Path Algorithm
    std::vector<int> predecessors(num_vertices(G));
    std::vector<float> distances(num_vertices(G), std::numeric_limits<float>::infinity());
    distances[init_state] = init_state; // Distance from source to source is 0
    bool success = bellman_ford_shortest_paths(G, weight_map(get(edge_weight, G))
        .predecessor_map(&predecessors[0])
        .distance_map(&distances[0]));

    int target_vertex = 2*T ; 
    int t1 = 0;
    int t2 = 0;
    float cost_p = 0.0;
    for (int v = target_vertex; v != init_state; v = predecessors[v]) { // Start from the target and go backwards
        t2 = (int)(v*0.5);
        t1 = (int)(predecessors[v]*0.5);
        cost_p += tab_cost[predecessors[v]][v];
        copy(sol[predecessors[v]][v].begin()+t1, sol[predecessors[v]][v].begin()+t2, solution.begin() + t1);
    }

    for(int t=0; t<T; t++){
        if (cf[t]<0 && solution[t]<x_on)
        {
            solution[t] = x_on;
        }
    }

    return;
}

vector<float> generate_cf(int T){
    // First create an instance of an engine.
    random_device rnd_device;
    // Specify the engine and distribution.
    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    uniform_int_distribution<int> dist {1, 52};
    
    auto gen = [&](){
                   return dist(mersenne_engine);
               };

    vector<float> vec(T);
    generate(vec.begin(), vec.end(), gen);

    return vec;
}



int main() {
    vector<int> tab_T = {24, 32, 40, 48, 56, 64, 72, 80, 88, 96};
    for (int T : tab_T) {
        float C_f = 10; // Constant C
        float x_on_f = 2.0; // Additional float
        vector<float> cf = { 0.7577430825087119, 0.34187950302620673, 0.32562054531596807, 0.9702299061199511, 0.4390826108558319, 0.6711465102155785, 0.6582497183772169, 0.23428318077603527, 0.9533748878363074, 0.0529404472909617, 0.4217112040306247, 0.13837880280408932, 0.38651206322038345, 0.2836440138880809, 0.34004011915005716, 0.7829042471052255, 0.343876986083249, 0.9530956202619553, 0.4162385940902089, 0.9182183320869829, 0.8291077500464802, 0.7619972259090257, 0.6871099356161865, 0.8898942762041043, 0.5026536471059008, 0.07725535726951815, 0.9493917827563235, 0.5981971995391653, 0.6034240292785478, 0.637005027453783, 0.20460103003742658, 0.41757868083199745, 0.6479258787044325, 0.8848260322773747, 0.6465707658723134, 0.43247144016369843, 0.4327752380149087, 0.7137234534088129, 0.2600377940149129, 0.09277526486316723, 0.5957420288930849, 0.6999092725845355, 0.2660878120815764, 0.980512418655227, 0.8081128341325331, 0.41006685300960977, 0.8565370641908052, 0.14933775788965498, 0.41469025618888744, 0.6148998140124149, 0.09951808141660112, 0.5883450372322966, 0.9608722338617036, 0.8878216902671419, 0.6102696915162856, 0.5751112519496402, 0.046552403014883725, 0.5346705664580781, 0.2779861118854451, 0.4859825276869111, 0.1677727931696794, 0.7251058260376907, 0.11656411996917981, 0.8367650858776531, 0.9911024171452134, 0.04703518639523785, 0.730488194669353, 0.4517189556277511, 0.09077086600313633, 0.09643325814878101, 0.7385555446468985, 0.7986523487161571, 0.642194468868843, 0.44629315311777207, 0.8133691483695563, 0.04624145829431947, 0.2564068779745857, 0.2895231034411342, 0.06465578398368332, 0.010749800961469047, 0.28661311584064275, 0.8082812013312853, 0.18293764332520823, 0.3071323290607493, 0.33358168316260206, 0.8439234927826565, 0.7881670609269731, 0.6836446889678165, 0.7560815309778708, 0.2951316412062436, 0.357860548833292, 0.5373513934872464, 0.2823322919675436, 0.05844667054480279, 0.5168487293848509, 0.49790565938523523, 0.7028609877400861, 0.01662245057194045, 0.6463713352735172, 0.7675705033570679, 0.71554457578748, 0.3435369304385266, 0.8877465725155745, 0.514292230310525, 0.990999463432171, 0.68039558994256, 0.23306221312090902, 0.19429293123886404, 0.6033651379578803, 0.6718258142012972, 0.12743541926608126, 0.04150298588551149, 0.9559653709963798, 0.5677921547910301, 0.3717596798851972, 0.6243995442568097, 0.1792743516185359, 0.15380976964723847, 0.484837503728619, 0.4966489919020538, 0.6102643595283643, 0.5494918918880075, 0.4446846294890686, 0.4877400326150223, 0.5012313643958933, 0.13501884956170485, 0.263682164814784, 0.870319700918931, 0.6540058962731771, 0.009929936206528156, 0.5892331106209994, 0.870071890767323, 0.5712992717142983, 0.3394026007550278, 0.5650955789924547, 0.8989215588447828, 0.9490116849651259, 0.41898810799514274, 0.6461533387921964, 0.023775720054089122, 0.05647156187851865, 0.6771410303244697, 0.7883411513629395, 0.2496607318224734, 0.42223312858239337, 0.44537733260746204, 0.82802955360734, 0.8678108928459863, 0.05315210337077014, 0.11122938214677391, 0.49059157342605475, 0.4722978052335315, 0.3325797769365534, 0.40264699630450995, 0.5379030084468349, 0.08183546310551548, 0.9663559937206737, 0.0026636347095423396, 0.17569778722606222, 0.7503508578714124, 0.35578839280701113, 0.6097286699848782, 0.50689002529537, 0.9229340109354833, 0.41809508617384705, 0.7053034643710893, 0.9460950268606492, 0.010756072928527316, 0.15408296251604425, 0.35464348188720396, 0.2697002078061955, 0.616890903024882, 0.28228324441146246, 0.8801179491195185, 0.34423418635989866, 0.46317803431421056, 0.4235615078159284, 0.9642745656295899, 0.270467452650778, 0.08221694166639482, 0.6094817218662998, 0.6398350841295396, 0.16443585848893094, 0.32843348263885874, 0.943078981504553, 0.7095327194226934, 0.36556707344289807, 0.603705770314298, 0.5734402515789646, 0.5160857597001575, 0.42695115206116296, 0.24487515540593374, 0.6251319088437705, 0.291121829300674, 0.9110114481554572, 0.18436917150167542, 0.3398993644085557, 0.99561032657725, 0.04947365617437616, 0.3475539132355445, 0.048851272663317014, 0.9221531860945376, 0.21492848906296858, 0.4773428343816054, 0.2357465100274051, 0.18814775085862945, 0.22340220990244908, 0.04426386911343583, 0.9394010158088115, 0.7661931985235821, 0.6184823943259136, 0.8934210720544925, 0.9419710298969195, 0.4925504529839707, 0.5877811042038525, 0.48434904803670153, 0.5513784716192107, 0.5082397588840005, 0.8805907769944604, 0.9433517006374261, 0.9321542028300817, 0.7351509279198581, 0.42123286725882814, 0.6519510485892882, 0.6768874875403861, 0.31978291928571057, 0.8142397924920192, 0.6858141899468986, 0.2881093860074945, 0.4810411638417804, 0.8031504558322925, 0.260144643149791, 0.3689036267910645, 0.6489310555259912, 0.17647934906534135, 0.3943629074551037, 0.7739632251108702, 0.25289078944336885, 0.30372645515876284, 0.6661854004451658, 0.10291733697126715, 0.09449986769615348, 0.15418004796356521, 0.912716560987844, 0.8428219149306857, 0.6388788694723904, 0.3171352335822174, 0.24613274863705892, 0.6808739219507934, 0.547062239752592, 0.37630984283319746, 0.9749705351784763, 0.2637827085385147, 0.05915948924757641, 0.643507669392707, 0.294769683010657, 0.6354768509674277, 0.6237463945128952, 0.7832208469209744, 0.8951794309399947, 0.40090295878422244, 0.6646449161606065, 0.42619757015423587, 0.5611318523305704, 0.07520893059126688, 0.3747135127043202, 0.40266868002032175, 0.5538442078665926, 0.660178809009167, 0.0904730951819892, 0.8880817351048655, 0.528991176450648, 0.6585434953297099, 0.1081711741275958, 0.11670596129621713, 0.8824107678959563, 0.3123137416452323, 0.6556413685138547, 0.7309000477498306, 0.7661882800256404, 0.7718581034742465, 0.4121520597812186, 0.9176902533431667, 0.6179838886077867, 0.7463641546339058, 0.13754796355060528, 0.8655591658239873, 0.5476408627055823, 0.11066056416513192, 0.41193369079741904, 0.3423178308242256, 0.6865539119957064, 0.35422958020334694, 0.15972706129920267, 0.9819022409957635, 0.09168129025923022, 0.9851381319141039, 0.6108244233792726, 0.3529286055182621, 0.03161616228428876, 0.41309922505945873, 0.12332294663942966, 0.02390869767968906, 0.7670086194422683, 0.173328073716, 0.984260022380037, 0.41326040635054084, 0.5577477795453762, 0.376884952631947, 0.007855427517530655, 0.5661648370071704, 0.5081668556262801, 0.2972799959195913, 0.844485161679963, 0.7195708732447852, 0.7204413110071353, 0.9348881616910523, 0.4162986290315036, 0.7091340511524348, 0.17293279051224253 };
        vector<float> d_f = { 2.12, 2.73, 2.58, 2.69, 5.63, 7.65, 1.36, 5.25, 0.2, 4.28, 9.51, 2.16, 8.58, 6.87, 8.93, 8.32, 3.12, 5.56, 2.64, 5.83, 6.26, 4.31, 9.83, 6.29, 5.05, 4.87, 0.96, 6.14, 0.93, 2.97, 1.16, 1.45, 8.02, 1.1, 1.65, 7.84, 1.74, 8.53, 1.84, 7.91, 7.76, 4.78, 8.83, 4.65, 5.4, 1.91, 9.83, 0.09, 7.99, 5.84, 8.44, 1.93, 4.28, 6.26, 3.05, 4.91, 3.54, 8.21, 8.5, 0.66, 5.76, 4.4, 6.72, 1.94, 7.7, 7.03, 5.76, 4.47, 0.97, 4.33, 8.91, 9.09, 6.22, 0.52, 4.12, 7.14, 7.23, 2.31, 8.49, 7.86, 7.17, 1.04, 0.57, 4.11, 9.49, 1.68, 0.87, 10.0, 6.88, 0.33, 6.22, 5.55, 6.72, 8.7, 3.1, 0.39, 3.85, 0.32, 3.24, 6.33, 4.03, 8.29, 8.94, 3.39, 6.02, 4.15, 9.7, 3.06, 9.85, 9.84, 8.13, 5.2, 3.73, 9.61, 5.87, 2.0, 7.19, 5.96, 4.38, 0.02, 1.13, 9.91, 3.58, 5.59, 2.4, 0.15, 7.89, 3.27, 2.42, 2.13, 0.41, 6.86, 5.09, 8.97, 9.06, 1.06, 7.56, 3.53, 1.96, 6.19, 5.18, 0.46, 8.74, 8.81, 3.14, 1.54, 5.7, 2.53, 1.36, 0.52, 8.78, 4.05, 0.83, 0.19, 0.86, 2.77, 7.46, 5.42, 4.3, 3.71, 7.78, 7.46, 9.12, 6.3, 0.73, 6.0, 8.67, 1.37, 9.71, 2.85, 2.73, 9.07, 3.93, 6.74, 7.16, 7.01, 6.86, 5.79, 9.07, 9.44, 5.12, 5.14, 1.69, 9.93, 9.17, 3.71, 3.84, 0.11, 2.68, 2.11, 6.08, 4.14, 7.46, 3.27, 9.64, 4.56, 1.25, 1.67, 0.09, 9.33, 0.07, 8.06, 9.29, 8.2, 7.42, 3.69, 9.33, 9.05, 4.2, 1.25, 2.67, 8.91, 7.39, 4.05, 5.97, 4.9, 9.24, 6.04, 8.49, 4.52, 4.04, 1.93, 4.55, 1.87, 7.24, 2.38, 2.49, 1.9, 6.35, 4.71, 7.81, 8.38, 0.37, 2.45, 0.94, 5.11, 6.98, 1.66, 9.33, 3.97, 6.1, 0.64, 2.01, 5.28, 9.0, 9.7, 0.12, 4.97, 6.8, 0.6, 4.44, 1.84, 3.44, 4.81, 8.96, 9.89, 8.16, 8.04, 8.12, 5.47, 5.6, 9.66, 7.29, 2.63, 4.81, 0.07, 5.45, 2.79, 2.94, 8.54, 2.35, 8.4, 8.41, 6.49, 9.16, 8.61, 3.84, 8.5, 7.37, 3.36, 5.41, 7.02, 0.54, 9.41, 0.75, 4.1, 4.81, 3.63, 3.76, 2.94, 6.59, 0.81, 4.02, 2.12, 4.73, 9.65, 1.37, 6.58, 6.96, 1.03, 8.99, 1.67, 1.37, 4.59, 3.19, 7.37, 0.65, 3.99, 9.45, 7.97, 3.44, 8.2, 0.36, 4.49, 3.71, 5.43, 8.67, 4.44, 7.46, 1.57 };
        vector<float> S_f = { 29.86, 26.22, 25.17, 27.42, 26.97, 25.91, 23.2, 28.39, 29.44, 26.37, 27.3, 29.59, 24.49, 28.98, 27.78, 21.7, 28.47, 23.41, 28.54, 22.89, 24.37, 21.76, 27.89, 28.78, 23.01, 23.4, 24.6, 20.86, 29.4, 26.55, 27.81, 28.65, 20.72, 21.28, 28.55, 29.03, 23.95, 24.52, 23.37, 23.52, 29.52, 29.74, 26.57, 29.92, 23.22, 25.56, 22.99, 29.64, 23.34, 22.13, 24.06, 22.81, 26.55, 22.66, 20.12, 21.83, 20.19, 20.51, 25.66, 24.83, 24.81, 25.71, 28.41, 27.01, 23.66, 26.0, 21.5, 25.68, 27.34, 21.73, 29.31, 24.67, 27.45, 24.88, 29.09, 27.66, 29.53, 24.27, 24.55, 28.17, 29.86, 22.55, 24.55, 22.58, 20.99, 25.51, 21.42, 20.36, 23.83, 22.54, 27.24, 25.26, 21.97, 27.46, 29.4, 23.86, 24.32, 29.03, 21.64, 21.82, 23.22, 21.83, 20.98, 23.49, 23.19, 23.93, 27.36, 21.29, 21.02, 21.72, 23.46, 29.71, 20.28, 20.53, 29.12, 29.97, 22.69, 20.48, 29.96, 29.53, 25.14, 25.14, 29.65, 24.97, 28.84, 26.48, 20.26, 25.7, 29.66, 24.86, 22.9, 20.1, 28.65, 25.05, 29.6, 26.66, 26.04, 20.04, 28.84, 29.29, 20.01, 22.56, 23.83, 20.12, 25.38, 24.44, 21.11, 27.11, 27.0, 22.93, 21.44, 24.05, 26.1, 22.85, 23.64, 24.76, 26.34, 21.44, 28.55, 21.67, 27.34, 22.17, 22.89, 21.19, 22.18, 22.06, 25.4, 28.31, 29.08, 23.23, 26.5, 23.79, 21.79, 24.05, 25.67, 29.67, 26.1, 25.85, 28.09, 26.48, 24.05, 29.67, 23.96, 26.33, 27.95, 28.82, 27.36, 21.51, 25.58, 21.82, 24.42, 27.34, 23.64, 25.99, 26.92, 25.41, 21.71, 24.09, 22.49, 25.21, 24.6, 24.86, 25.25, 26.3, 21.07, 21.78, 24.24, 26.31, 28.41, 22.36, 21.91, 21.43, 27.02, 29.13, 27.28, 28.38, 21.15, 28.12, 22.91, 26.47, 23.34, 29.68, 20.55, 25.06, 29.49, 25.07, 23.16, 22.23, 25.91, 25.93, 27.76, 24.37, 23.46, 28.8, 23.39, 28.33, 23.42, 20.94, 27.26, 26.62, 23.51, 28.34, 29.91, 26.16, 24.19, 29.96, 21.39, 26.13, 26.41, 27.1, 27.5, 26.03, 24.52, 25.04, 25.43, 22.34, 24.48, 20.17, 26.31, 21.86, 23.21, 26.9, 28.59, 22.46, 25.82, 22.45, 27.48, 29.89, 27.85, 24.92, 29.18, 22.51, 26.4, 22.18, 28.67, 24.63, 25.07, 25.61, 22.79, 27.78, 26.42, 28.15, 28.83, 27.78, 27.12, 23.17, 20.64, 21.54, 22.22, 22.98, 27.23, 25.81, 22.96, 20.87, 22.58, 22.71, 20.77, 21.54, 27.27, 23.71, 23.11, 28.84, 26.76, 25.88, 24.5, 26.92, 20.89, 27.7, 22.39, 25.09, 24.36, 20.66, 21.92, 27.65, 26.4, 27.05, 20.05, 22.05, 23.38, 20.3 };    //vector<float> S_f = vector<float>(T, 21);

        int s_in = 0;                // Stock in coefficient
        int s_out = 0;               // Stock out coefficient
        float s_i_f = 0.0;            // Initial stock value

        //float to int
        int factor = pow(10, int(log(1. / (C_f * 1e-5)) * 1. / log(10)));
        int C = int(C_f * factor);
        int x_on = int(x_on_f * factor);
        int s_i = int(s_i_f * factor);
        vector<int> d(T, 0);
        vector<int> S(T, 0);
        for (int t = 0; t < T; ++t) {
            d[t] = int(d_f[t] * factor);
            S[t] = int(S_f[t] * factor * 4);
        }

        normalize_ini_stock(d, S, T, s_i);
        normalize_demand(d, S, T, C);
        normalize_bounds(d, S, T, C);

        vector<int> res(T, 0);
        auto start = high_resolution_clock::now();
        CLSP_T3(res, T, cf, S, d, C, x_on);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << duration.count() << endl;

        //Int to float
        vector<float> sol(T, 0.0);
        for (int t = 0; t < T; ++t) {
            sol[t] = res[t] * 1. / factor;
        }

        float I_f = s_i_f;
        for (int t = 0; t < T; ++t) {
            I_f += sol[t] - d_f[t];
        }

        float cost_f = 0;
        for (int t = 0; t < T; ++t) {
            if (sol[t] > 0) {
                sol[t] += x_on_f;
            }
            cost_f += sol[t] * cf[t];
        }

        printf("\n Cost: %f \n", cost_f);

    }
    return 0;
}

