#include <bits/stdc++.h>
#include "testlib.h"

using namespace std;

int main(int argc, char *argv[]) {
    registerValidation(argc, argv);

    int t = inf.readInt(1, 10000, "t");
    inf.readEoln();

    long long total_n = 0;
    long long total_q = 0;

    for (int tc = 0; tc < t; ++tc) {
        int n = inf.readInt(1, 200000, "n");
        inf.readEoln();
        total_n += n;
        ensuref(total_n <= 200000, "Sum of n over all test cases exceeds 200000");

        vector<char> seen(n + 1, 0);
        for (int i = 1; i <= n; ++i) {
            int pi = inf.readInt(1, n, "p_i");
            ensuref(!seen[pi], "Duplicate value %d in permutation (test case %d)", pi, tc + 1);
            seen[pi] = 1;
            if (i < n) inf.readSpace(); else inf.readEoln();
        }

        string s = inf.readToken("[01]+", "s");
        ensuref((int)s.size() == n, "String s length %d does not match n=%d (test case %d)", (int)s.size(), n, tc + 1);
        inf.readEoln();

        int q = inf.readInt(1, 200000, "q");
        inf.readEoln();
        total_q += q;
        ensuref(total_q <= 200000, "Sum of q over all test cases exceeds 200000");

        for (int i = 0; i < q; ++i) {
            int v = inf.readInt(1, n, "v");
            inf.readSpace();
            long long k = inf.readLong(0, 1000000000000000000LL, "k");
            inf.readEoln();
        }
    }

    inf.readEof();
    return 0;
}