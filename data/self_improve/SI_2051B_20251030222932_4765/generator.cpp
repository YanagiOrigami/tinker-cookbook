//  Generator for the “Journey with a Modulo Condition” problem.
//  Usage:  ./gen <mode>   (mode is an integer from 1 to 10)
//  The generator writes a test to stdout according to the chosen strategy.

#include <bits/stdc++.h>
using namespace std;

using int64 = long long;
const int64 MAX_N = (int64)1e18;
const int   MAX_ABC = 1000000000;
const int   MAX_P   = 1000000000;
const int   MAX_T   = 10000;

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

int64 rnd64(int64 l, int64 r) {
    uniform_int_distribution<int64> d(l, r);
    return d(rng);
}
int rndint(int l, int r) {
    uniform_int_distribution<int> d(l, r);
    return d(rng);
}

/* --------------------------------------------------------------- */
/* 1. t = MAX_T, t‑1 minimal cases, one maximal n                  */
void mode1() {
    cout << MAX_T << '\n';
    for (int i = 0; i < MAX_T - 1; ++i)
        cout << 1 << ' ' << 1 << ' ' << 1 << ' ' << 1 << ' ' << 1 << '\n';
    cout << MAX_N << ' ' << 1 << ' ' << 1 << ' ' << 1 << ' ' << 1 << '\n';
}

/* --------------------------------------------------------------- */
/* 2. t = MAX_T, all cases have the same large n                  */
void mode2() {
    cout << MAX_T << '\n';
    for (int i = 0; i < MAX_T; ++i) {
        int a = rndint(1, MAX_ABC);
        int b = rndint(1, MAX_ABC);
        int c = rndint(1, MAX_ABC);
        int p = rndint(1, MAX_P);
        cout << MAX_N << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
    }
}

/* --------------------------------------------------------------- */
/* 3. Produce cases where the answer D is huge (≈1e18)            */
void mode3() {
    const int t = 1000;
    cout << t << '\n';
    for (int i = 0; i < t; ++i) {
        // a=b=c=1 makes the cumulative sum equal to the day count.
        // Choose a large p so the first multiple ≥ n is close to n.
        int p = MAX_P;                 // 1e9
        cout << MAX_N << ' ' << 1 << ' ' << 1 << ' ' << 1 << ' ' << p << '\n';
    }
}

/* --------------------------------------------------------------- */
/* 4. Long contiguous sequence of identical values                */
void mode4() {
    cout << MAX_T << '\n';
    int64 n = 123456789012345LL;
    int a = 7, b = 7, c = 7, p = 13;
    for (int i = 0; i < MAX_T; ++i)
        cout << n << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
}

/* --------------------------------------------------------------- */
/* 5. Strictly increasing sequence across test cases             */
void mode5() {
    cout << MAX_T << '\n';
    int64 base = 1;
    for (int i = 0; i < MAX_T; ++i) {
        int64 n = base + i * 1000LL;           // strictly increasing
        int a = 1, b = 1, c = 1, p = 2;
        cout << n << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
    }
}

/* --------------------------------------------------------------- */
/* 6. Strictly decreasing sequence across test cases             */
void mode6() {
    cout << MAX_T << '\n';
    int64 start = 1000000000000LL; // 1e12
    for (int i = 0; i < MAX_T; ++i) {
        int64 n = start - i * 1000LL;   // strictly decreasing, stays >0
        int a = 1, b = 1, c = 1, p = 2;
        cout << n << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
    }
}

/* --------------------------------------------------------------- */
/* 7. Alternating large and small values                           */
void mode7() {
    cout << MAX_T << '\n';
    for (int i = 0; i < MAX_T; ++i) {
        if (i % 2 == 0) {
            // large case
            int a = rndint(1, MAX_ABC);
            int b = rndint(1, MAX_ABC);
            int c = rndint(1, MAX_ABC);
            int p = rndint(1, MAX_P);
            cout << MAX_N << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
        } else {
            // tiny case
            cout << 1 << ' ' << 1 << ' ' << 1 << ' ' << 1 << ' ' << 1 << '\n';
        }
    }
}

/* --------------------------------------------------------------- */
/* 8. Worst‑case for brute‑force (simulate day by day)             */
void mode8() {
    // One huge case where D = 1e18 (needs 1e18 iterations for a naive loop)
    cout << 1 << '\n';
    int a = 1, b = 1, c = 1;
    int p = MAX_P;                 // 1e9
    cout << MAX_N << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
}

/* --------------------------------------------------------------- */
/* 9. Problem‑specific: maximize answer D with prime modulo      */
void mode9() {
    const int t = 5;
    cout << t << '\n';
    int64 n = MAX_N - 5;           // slightly below 1e18
    int a = 1, b = 1, c = 1;
    int p = 999999937;             // a large prime ≤ 1e9
    for (int i = 0; i < t; ++i)
        cout << n << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
}

/* --------------------------------------------------------------- */
/* 10. Edge conditions (p = 1, maximal a,b,c)                     */
void mode10() {
    const int t = 5;
    cout << t << '\n';
    int64 n = MAX_N;
    int a = MAX_ABC, b = MAX_ABC, c = MAX_ABC;
    int p = 1;                     // always divisible
    for (int i = 0; i < t; ++i)
        cout << n << ' ' << a << ' ' << b << ' ' << c << ' ' << p << '\n';
}

/* --------------------------------------------------------------- */
int main(int argc, char* argv[]) {
    if (argc != 2) return 0;
    int mode = stoi(argv[1]);
    switch (mode) {
        case 1: mode1(); break;
        case 2: mode2(); break;
        case 3: mode3(); break;
        case 4: mode4(); break;
        case 5: mode5(); break;
        case 6: mode6(); break;
        case 7: mode7(); break;
        case 8: mode8(); break;
        case 9: mode9(); break;
        case 10: mode10(); break;
        default: return 0;
    }
    return 0;
}