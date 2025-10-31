//  Test generator for the "Black Nodes in Prefix of a Permutation Cycle" problem.
//  Generates ten different kinds of tests, selected by a command line argument
//  (1 … 10).  All generated data obey the limits of the original problem.
//
//  Compile with:  g++ -std=c++17 -O2 -pipe -static -s -o generator generator.cpp -ltestlib
//  Run with:      ./generator <type> > test.in
//
//  The generator uses the testlib library (https://github.com/MikeMirzayanov/testlib).

#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_T = 10000;          // maximal number of test cases
constexpr int MAX_N_SUM = 200000;     // Σ n over all test cases
constexpr int MAX_Q_SUM = 200000;     // Σ q over all test cases

/*---------------------------------------------------------------*/
/* Utility to output one test case                               */
static void output_case(int n, const vector<int>& p,
                        const string& s, int q,
                        const vector<pair<int64_t,int64_t>>& queries)
{
    cout << n << "\n";
    for (int i = 0; i < n; ++i) {
        if (i) cout << ' ';
        cout << p[i];
    }
    cout << "\n";
    cout << s << "\n";
    cout << q << "\n";
    for (auto [v, k] : queries) {
        cout << v << ' ' << k << "\n";
    }
}

/*---------------------------------------------------------------*/
/* 1. t = MAX_T, (t‑1) cases tiny (n = 1), one case huge           */
static void gen_type1()
{
    int t = MAX_T;
    cout << t << "\n";

    // small cases
    for (int i = 0; i < t-1; ++i) {
        int n = 1;
        vector<int> p = {1};
        string s = "0";               // black, makes answer non‑zero
        int q = 1;
        vector<pair<int64_t,int64_t>> queries = {{1, 0}};
        output_case(n, p, s, q, queries);
    }

    // one large case – fill the remaining budget
    int used_n = (t-1) * 1;
    int n_big = MAX_N_SUM - used_n;          // ≤ 200000
    vector<int> p_big(n_big);
    for (int i = 0; i < n_big; ++i) p_big[i] = i+1;   // identity (single cycle)

    string s_big(n_big, '0');                // all black → answers maximal
    int used_q = (t-1) * 1;
    int q_big = MAX_Q_SUM - used_q;          // use the rest of the query budget

    vector<pair<int64_t,int64_t>> queries_big;
    queries_big.reserve(q_big);
    for (int i = 0; i < q_big; ++i) {
        int v = 1 + (i % n_big);
        int64_t k = 1000000000000000000LL;   // 1e18, the maximal allowed
        queries_big.emplace_back(v, k);
    }
    output_case(n_big, p_big, s_big, q_big, queries_big);
}

/*---------------------------------------------------------------*/
/* 2. t = MAX_T, all cases have the same large n (≈20)           */
static void gen_type2()
{
    int t = MAX_T;
    int n_each = MAX_N_SUM / t;               // exactly 20 when MAX_T = 10000
    cout << t << "\n";

    for (int tc = 0; tc < t; ++tc) {
        vector<int> p(n_each);
        for (int i = 0; i < n_each; ++i) p[i] = i+1; // identity

        string s(n_each, rnd.next(0,1) ? '1' : '0');

        int q = n_each;                       // use the whole per‑case budget
        vector<pair<int64_t,int64_t>> queries;
        for (int i = 0; i < q; ++i) {
            int v = 1 + rnd.next(0, n_each-1);
            int64_t k = rnd.next(0LL, 1000LL); // moderate k
            queries.emplace_back(v, k);
        }
        output_case(n_each, p, s, q, queries);
    }
}

/*---------------------------------------------------------------*/
/* 3. Single case, output size maximised (all zeros, huge k)    */
static void gen_type3()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i+1;   // identity

    string s(n, '0');                         // every position black

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    queries.reserve(q);
    for (int i = 0; i < q; ++i) {
        int v = 1 + rnd.next(0, n-1);
        int64_t k = 1000000000000000000LL;    // 1e18
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 4. Long contiguous identical values – string of all '1's    */
static void gen_type4()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = (i+2 <= n) ? i+2 : 1; // single big cycle

    string s(n, '1');                         // all white, identical values

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int v = 1;
        int64_t k = i;                        // increasing k, still simple
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 5. Strictly increasing permutation (identity)               */
static void gen_type5()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p(n);
    iota(p.begin(), p.end(), 1);              // 1,2,3,…,n

    string s(n, '0');
    for (int i = 0; i < n; ++i) s[i] = (i%2) ? '1' : '0'; // alternating colours

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int v = 1 + (i % n);
        int64_t k = rnd.next(0LL, 1000000LL);
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 6. Strictly decreasing permutation                           */
static void gen_type6()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = n - i; // n, n-1, … ,1

    string s(n, '0');                         // all black

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int v = 1 + rnd.next(0, n-1);
        int64_t k = rnd.next(0LL, 5000LL);
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 7. Alternating large / small values in permutation           */
static void gen_type7()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p;
    p.reserve(n);
    int l = 1, r = n;
    while ((int)p.size() < n) {
        p.push_back(l);
        ++l;
        if ((int)p.size() < n) {
            p.push_back(r);
            --r;
        }
    }

    string s(n, '0');
    for (int i = 0; i < n; ++i)
        if (i % 3 == 0) s[i] = '1';

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int v = 1 + rnd.next(0, n-1);
        int64_t k = rnd.next(0LL, 10000LL);
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 8. Worst‑case for naive O(k) simulation (huge k, long cycle)  */
static void gen_type8()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p(n);
    // create a single cycle of length n
    for (int i = 0; i < n-1; ++i) p[i] = i+2;
    p[n-1] = 1;

    string s(n, '0');                         // all black → each step counts

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int v = 1 + rnd.next(0, n-1);
        int64_t k = 1000000000000000000LL;    // 1e18, forces O(k) algorithm to TLE
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 9. Problem‑specific extreme: many cycles, all black           */
static void gen_type9()
{
    int t = 1;
    cout << t << "\n";

    int n = MAX_N_SUM;
    vector<int> p(n);
    // build cycles of length 2 (or 1 if n is odd)
    for (int i = 0; i+1 < n; i += 2) {
        p[i] = i+2;
        p[i+1] = i+1;
    }
    if (n % 2) p[n-1] = n; // last element stays a self‑loop

    string s(n, '0');                         // every vertex black

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int v = 1 + rnd.next(0, n-1);
        int64_t k = rnd.next(0LL, 1000LL);
        queries.emplace_back(v, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
/* 10. Edge‑case heavy: n = 1, huge number of queries, k up to 1e18 */
static void gen_type10()
{
    int t = 1;
    cout << t << "\n";

    int n = 1;
    vector<int> p = {1};
    string s = "0";               // black, answer = min(k+1,1)

    int q = MAX_Q_SUM;
    vector<pair<int64_t,int64_t>> queries;
    for (int i = 0; i < q; ++i) {
        int64_t k = (i % 2 == 0) ? 0 : 1000000000000000000LL;
        queries.emplace_back(1, k);
    }
    output_case(n, p, s, q, queries);
}

/*---------------------------------------------------------------*/
int main(int argc, char* argv[])
{
    registerGen(argc, argv, 1);
    int type = atoi(argv[1]);
    // seed based on test number to get reproducible random data
    rnd.setSeed(chrono::high_resolution_clock::now().time_since_epoch().count());

    switch (type) {
        case 1: gen_type1(); break;
        case 2: gen_type2(); break;
        case 3: gen_type3(); break;
        case 4: gen_type4(); break;
        case 5: gen_type5(); break;
        case 6: gen_type6(); break;
        case 7: gen_type7(); break;
        case 8: gen_type8(); break;
        case 9: gen_type9(); break;
        case 10: gen_type10(); break;
        default: {
            cerr << "Invalid type: " << type << ". Provide a number from 1 to 10.\n";
            return 1;
        }
    }
    return 0;
}