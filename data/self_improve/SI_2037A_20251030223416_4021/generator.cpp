//  Generator for "Non‑Crossing Equal Pairs"
//  Compile with:  g++ -std=c++17 -O2 -Wall -Wextra generator.cpp -o generator
//  Usage: ./generator <type>
//  where <type> is an integer from 1 to 10 selecting the test‑generation strategy.

#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <type (1..10)>\n";
        return 0;
    }
    int type = atoi(argv[1]);
    const int MAX_T = 10;
    const int MAX_N = 500;

    // helper to print one test case
    auto print_case = [&](int n, const vector<int>& a) {
        cout << n << "\n";
        for (int i = 0; i < n; ++i) {
            if (i) cout << ' ';
            cout << a[i];
        }
        cout << "\n";
    };

    // -----------------------------------------------------------------
    // Strategy implementations
    // -----------------------------------------------------------------
    if (type == 1) {
        // t = 10, 9 cases minimal (n=1), 1 case maximal (n=500)
        cout << MAX_T << "\n";
        for (int i = 0; i < MAX_T - 1; ++i) {
            print_case(1, {1});
        }
        // large case – random values
        vector<int> a(MAX_N);
        for (int i = 0; i < MAX_N; ++i) a[i] = rnd.next(1, MAX_N);
        print_case(MAX_N, a);
    }
    else if (type == 2) {
        // t = 10, every case large (n = 500) – random values
        cout << MAX_T << "\n";
        for (int tc = 0; tc < MAX_T; ++tc) {
            vector<int> a(MAX_N);
            for (int i = 0; i < MAX_N; ++i) a[i] = rnd.next(1, MAX_N);
            print_case(MAX_N, a);
        }
    }
    else if (type == 3) {
        // t = 10, each case designed to give maximal answer: all elements equal
        cout << MAX_T << "\n";
        vector<int> a(MAX_N, 1);          // all 1's → floor(500/2) pairs possible
        for (int tc = 0; tc < MAX_T; ++tc) {
            print_case(MAX_N, a);
        }
    }
    else if (type == 4) {
        // Single test: long contiguous block of identical values
        cout << 1 << "\n";
        vector<int> a(MAX_N, 42);         // any value between 1 and n, e.g., 42
        print_case(MAX_N, a);
    }
    else if (type == 5) {
        // Single test: strictly increasing sequence 1..n
        cout << 1 << "\n";
        vector<int> a(MAX_N);
        for (int i = 0; i < MAX_N; ++i) a[i] = i + 1;
        print_case(MAX_N, a);
    }
    else if (type == 6) {
        // Single test: strictly decreasing sequence n..1
        cout << 1 << "\n";
        vector<int> a(MAX_N);
        for (int i = 0; i < MAX_N; ++i) a[i] = MAX_N - i;
        print_case(MAX_N, a);
    }
    else if (type == 7) {
        // Single test: alternating small / large values
        cout << 1 << "\n";
        vector<int> a;
        a.reserve(MAX_N);
        int small = 1, large = MAX_N;
        for (int i = 0; i < MAX_N; ++i) {
            if (i % 2 == 0) a.push_back(small++);   // 1,2,3,...
            else          a.push_back(large--);    // n,n-1,n-2,...
        }
        print_case(MAX_N, a);
    }
    else if (type == 8) {
        // Single test: worst‑case for naive O(n³) DP – many crossing equal pairs
        // Build: 1 2 3 … 250 1 2 3 … 250
        cout << 1 << "\n";
        const int half = MAX_N / 2;           // 250
        vector<int> a;
        a.reserve(MAX_N);
        for (int v = 1; v <= half; ++v) a.push_back(v);
        for (int v = 1; v <= half; ++v) a.push_back(v);
        print_case(MAX_N, a);
    }
    else if (type == 9) {
        // Single test: each value appears exactly twice in nested order
        // 1 2 3 … 250 250 … 3 2 1
        cout << 1 << "\n";
        const int half = MAX_N / 2;           // 250
        vector<int> a;
        a.reserve(MAX_N);
        for (int v = 1; v <= half; ++v) a.push_back(v);        // left side
        for (int v = half; v >= 1; --v) a.push_back(v);        // right side (nested)
        print_case(MAX_N, a);
    }
    else if (type == 10) {
        // Single test: edge condition – only one possible pair far apart
        // a1 = aN = 1, all middle elements distinct
        cout << 1 << "\n";
        vector<int> a(MAX_N, 0);
        a[0] = a[MAX_N - 1] = 1;
        int cur = 2;
        for (int i = 1; i < MAX_N - 1; ++i) {
            a[i] = cur;
            ++cur;
            if (cur > MAX_N) cur = 2;               // wrap, but still ≤ n
        }
        print_case(MAX_N, a);
    }
    else {
        // Unknown type – fallback to a random small test
        cerr << "Unknown type " << type << ". Generating a random test.\n";
        int t = rnd.next(1, MAX_T);
        cout << t << "\n";
        for (int tc = 0; tc < t; ++tc) {
            int n = rnd.next(1, MAX_N);
            vector<int> a(n);
            for (int i = 0; i < n; ++i) a[i] = rnd.next(1, n);
            print_case(n, a);
        }
    }
    return 0;
}