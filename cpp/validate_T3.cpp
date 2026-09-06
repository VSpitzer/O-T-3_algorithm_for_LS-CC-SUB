#define main clsp_unused_main
#include "solve_CLSP_IB.cpp"
#undef main
#include <cstdint>
#include <random>
using std::vector;
static const long long BIG = (long long)4e18;

// exact DP + optimal plan reconstruction
static long long exact_dp(int T, const vector<int>& d, const vector<int>& S, int C, int x_on,
                          const vector<int>& cfi, vector<int>* plan) {
    int Smax = 0; for (int t = 0; t < T; ++t) Smax = std::max(Smax, S[t]);
    vector<vector<long long>> dp(T + 1, vector<long long>(Smax + 1, BIG));
    vector<vector<int>> ch(T + 1, vector<int>(Smax + 1, -1));
    dp[0][0] = 0;
    for (int t = 0; t < T; ++t)
        for (int Ip = 0; Ip <= Smax; ++Ip) {
            if (dp[t][Ip] == BIG) continue;
            for (int x = 0; x <= C; ++x) {
                int I = Ip + x - d[t];
                if (I < 0 || I > S[t]) continue;
                long long c = dp[t][Ip] + (long long)cfi[t] * (x + (x > 0 ? x_on : 0));
                if (c < dp[t+1][I]) { dp[t+1][I] = c; ch[t+1][I] = x; }
            }
        }
    if (plan) { plan->assign(T, 0); int I = 0;
        for (int t = T; t >= 1; --t) { int x = ch[t][I]; (*plan)[t-1] = x; I = I - x + d[t-1]; } }
    return dp[T][0];
}

int main(int argc, char** argv) {
    const int C = 10; const int x_on = (argc>6)?atoi(argv[6]):2;
    const int NPER = (argc > 1) ? atoi(argv[1]) : 100;
    const int Slo  = (argc > 2) ? atoi(argv[2]) : 2;     // raw S in [Slo*C, Shi*C]
    const int Shi  = (argc > 3) ? atoi(argv[3]) : 4;
    std::mt19937 rng((argc > 4) ? atoi(argv[4]) : 999);
    printf("raw S in [%dC,%dC]; %d instances per T\n", Slo, Shi, NPER);
    long long histo_dev[16] = {0}, histo_all[16] = {0};

    for (int T : {110, 120, 150}) {
        int bad = 0; long long sumexcess = 0;
        for (int it = 0; it < NPER; ++it) {
            vector<int> d(T), S(T), cfi(T);
            std::uniform_int_distribution<int> dd(0,C), dS(Slo*C,Shi*C), dc(1,100);
            for (int t = 0; t < T; ++t){ d[t]=dd(rng); S[t]=dS(rng); cfi[t]=dc(rng); }
            vector<float> cf(T); for (int t=0;t<T;++t) cf[t]=(float)cfi[t];
            vector<int> dn=d, Sn=S;
            normalize_ini_stock(dn,Sn,T,0); normalize_demand(dn,Sn,T,C); normalize_bounds(dn,Sn,T,C);
            vector<int> below; for (int t=0;t<T;++t) if (Sn[t] < 2*C) below.push_back(t);
            int nb = (int)below.size(); if (nb<16) histo_all[nb]++;
            int lastbelow = below.empty() ? -1 : below.back();

            vector<int> res(T,0); CLSP_T3(res,T,cf,Sn,dn,C,x_on);
            long long t3=0; for (int t=0;t<T;++t) t3 += (long long)cfi[t]*(res[t]+(res[t]>0?x_on:0));
            vector<int> opl; long long opt = exact_dp(T,dn,Sn,C,x_on,cfi,&opl);
            if (t3 != opt) {
                ++bad; sumexcess += t3-opt; if (nb<16) histo_dev[nb]++;
                int firstdiff=-1; for (int t=0;t<T;++t) if (res[t]!=opl[t]) { firstdiff=t; break; }
                // where does the cost excess accumulate?
                long long c3=0,co=0; int divergefrom=-1;
                for (int t=0;t<T;++t){ c3+=(long long)cfi[t]*(res[t]+(res[t]>0?x_on:0));
                                       co+=(long long)cfi[t]*(opl[t]+(opl[t]>0?x_on:0));
                                       if (divergefrom<0 && c3!=co) divergefrom=t; }
                if (bad<=8) printf("  T=%d #%d  excess=%+lld  first plan diff at t=%d, cost diverges at t=%d"
                                   "  | S<2C at periods {%s%d..%d} last=%d\n",
                                   T,it,t3-opt,firstdiff,divergefrom, nb?"":"none", nb?below.front():-1,
                                   nb?below.back():-1, lastbelow);
            }
        }
        printf("T=%-4d -> %d/%d deviate (all positive), total excess=%lld\n", T,bad,NPER,sumexcess);
    }
    printf("\n#periods with S<2C : instances (all) / (deviating)\n");
    for (int k=0;k<16;k++) if (histo_all[k]) printf("  %2d : %5lld / %lld\n", k, histo_all[k], histo_dev[k]);
    return 0;
}
