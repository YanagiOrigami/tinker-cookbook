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
        total_n += n;
        inf.readEoln();

        vector<char> used(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            int x = inf.readInt(1, n, "p_i");
            ensuref(!used[x],
                    "Duplicate value %d in permutation (test %d, position %d)",
                    x, tc + 1, i + 1);
            used[x] = 1;
            if (i + 1 < n) inf.readSpace();
            else inf.readEoln();
        }

        string s = inf.readToken("[01]+", "s");
        ensuref((int)s.size() == n,
                "Binary string length %d does not match n=%d (test %d)",
                (int)s.size(), n, tc + 1);
        inf.readEoln();

        int q = inf.readInt(1, 200000, "q");
        total_q += q;
        inf.readEoln();

        for (int i = 0; i < q; ++i) {
            int v = inf.readInt(1, n, "v");
            inf.readSpace();
            long long k = inf.readLong(0, 1000000000000000000LL, "k");
            inf.readEoln();
        }
    }

    ensuref(total_n <= 200000,
            "Sum of n over all test cases %lld exceeds limit 200000",
            total_n);
    ensuref(total_q <= 200000,
            "Sum of q over all test cases %lld exceeds limit 200000",
            total_q);

    inf.readEof();
    return 0;
}