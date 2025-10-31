#include <bits/stdc++.h>
#include "testlib.h"

using namespace std;

int main(int argc, char *argv[]) {
    registerValidation(argc, argv);

    const long long N_MIN = 1LL;
    const long long N_MAX = 1000000000000000000LL;          // 1e18
    const int ABc_MIN = 1;
    const int ABc_MAX = 1000000000;                        // 1e9
    const int P_MIN = 1;
    const int P_MAX = 1000000000;                          // 1e9
    const int T_MIN = 1;
    const int T_MAX = 10000;                               // 1e4

    int t = inf.readInt(T_MIN, T_MAX, "t");
    inf.readEoln();

    for (int tc = 0; tc < t; ++tc) {
        long long n = inf.readLong(N_MIN, N_MAX, "n");
        inf.readSpace();

        int a = inf.readInt(ABc_MIN, ABc_MAX, "a");
        inf.readSpace();

        int b = inf.readInt(ABc_MIN, ABc_MAX, "b");
        inf.readSpace();

        int c = inf.readInt(ABc_MIN, ABc_MAX, "c");
        inf.readSpace();

        int p = inf.readInt(P_MIN, P_MAX, "p");
        inf.readEoln();
    }

    inf.readEof();
    return 0;
}