//  Alternating String Reconstruction – test generator
//  Compile with:  g++ -std=c++17 -O2 -Wall -Wextra generator.cpp -o generator
//  Usage: ./generator <type>
//
//  <type> is an integer from 1 to 10 selecting the generation strategy.

#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

string rand_str(int n) {
    string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i) {
        s.push_back('a' + rnd.next(0, 25));
    }
    return s;
}

// strategy 1 : max t, (t‑1) cases minimal, one case maximal
void gen1() {
    const int MAX_T = 10000;
    const int MAX_N = 200000;
    int t = MAX_T;
    int small_n = 1;
    int big_n = MAX_N - (t - 1) * small_n;          // = 190001

    cout << t << "\n";
    for (int i = 0; i < t - 1; ++i) {
        cout << small_n << "\n";
        cout << rand_str(small_n) << "\n";
    }
    cout << big_n << "\n";
    cout << rand_str(big_n) << "\n";
}

// strategy 2 : max t, all cases same large n, total size maximal
void gen2() {
    const int MAX_T = 10000;
    const int MAX_N = 200000;
    int t = MAX_T;
    int n = MAX_N / t;               // 20
    cout << t << "\n";
    for (int i = 0; i < t; ++i) {
        cout << n << "\n";
        cout << rand_str(n) << "\n";
    }
}

// strategy 3 : maximise the answer (cost) for each case
void gen3() {
    const int n = 200000;
    cout << 1 << "\n";
    cout << n << "\n";
    cout << rand_str(n) << "\n";
}

// strategy 4 : long block of identical letters
void gen4() {
    const int n = 200000;
    cout << 1 << "\n";
    cout << n << "\n";
    cout << string(n, 'a') << "\n";
}

// strategy 5 : strictly increasing sequence (max possible length = 26)
void gen5() {
    const string inc = "abcdefghijklmnopqrstuvwxyz";
    cout << 1 << "\n";
    cout << (int)inc.size() << "\n";
    cout << inc << "\n";
}

// strategy 6 : strictly decreasing sequence (max possible length = 26)
void gen6() {
    const string dec = "zyxwvutsrqponmlkjihgfedcba";
    cout << 1 << "\n";
    cout << (int)dec.size() << "\n";
    cout << dec << "\n";
}

// strategy 7 : alternating large and small values (a and z)
void gen7() {
    const int n = 200000;
    string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
        s.push_back(i % 2 ? 'z' : 'a');   // a z a z ...
    cout << 1 << "\n";
    cout << n << "\n";
    cout << s << "\n";
}

// strategy 8 : worst‑case for naive O(26²) approaches – perfectly balanced frequencies
void gen8() {
    const int n = 200000;
    string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i)
        s.push_back('a' + (i % 26));
    cout << 1 << "\n";
    cout << n << "\n";
    cout << s << "\n";
}

// strategy 9 : problem‑specific – minimise max frequency on even/odd positions
void gen9() {
    const int n = 200000;
    string s;
    s.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0)               // even position
            s.push_back('a' + (i / 2 % 26));
        else                          // odd position, shift by 13 to avoid overlap
            s.push_back('a' + ((i / 2 + 13) % 26));
    }
    cout << 1 << "\n";
    cout << n << "\n";
    cout << s << "\n";
}

// strategy 10 : edge cases – mix of tiny strings and one large one
void gen10() {
    vector<string> tiny = {
        "a",                // n=1
        "ab",               // n=2 already alternating
        "aab",              // n=3 needs deletion
        "abcd",             // n=4 random
        "abcde"             // n=5 odd length
    };
    const int big_n = 190000;               // leaves room for the tiny cases (total ≤200k)
    vector<string> cases;
    for (auto &x : tiny) cases.push_back(x);
    cases.push_back(rand_str(big_n));

    cout << (int)cases.size() << "\n";
    for (auto &s : cases) {
        cout << (int)s.size() << "\n";
        cout << s << "\n";
    }
}

int main(int argc, char * argv[]) {
    registerGen(argc, argv, 1);
    int type = argc > 1 ? atoi(argv[1]) : 1;
    // ensure type is between 1 and 10
    if (type < 1) type = 1;
    if (type > 10) type = 10;

    // initialise random generator
    rnd.setSeed(rnd.next());

    switch (type) {
        case 1: gen1(); break;
        case 2: gen2(); break;
        case 3: gen3(); break;
        case 4: gen4(); break;
        case 5: gen5(); break;
        case 6: gen6(); break;
        case 7: gen7(); break;
        case 8: gen8(); break;
        case 9: gen9(); break;
        case 10: gen10(); break;
        default: gen1(); break;
    }
    return 0;
}