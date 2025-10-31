#include <bits/stdc++.h>
#include "testlib.h"

using namespace std;

int main(int argc, char *argv[]) {
    registerValidation(argc, argv);

    int t = inf.readInt(1, 10000, "t");
    inf.readEoln();

    const long long N_MAX = 1000000000000000000LL;          // 1e18
    const int LIM_ABCP = 1000000000;                       // 1e9

    for (int i = 0; i < t; ++i) {
        long long n = inf.readLong(1, N_MAX, "n");
        inf.readSpace();
        int a = inf.readInt(1, LIM_ABCP, "a");
        inf.readSpace();
        int b = inf.readInt(1, LIM_ABCP, "b");
        inf.readSpace();
        int c = inf.readInt(1, LIM_ABCP, "c");
        inf.readSpace();
        int p = inf.readInt(1, LIM_ABCP, "p");
        inf.readEoln();
    }

    inf.readEof();
    return 0;
}