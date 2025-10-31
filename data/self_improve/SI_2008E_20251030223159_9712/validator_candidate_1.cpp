#include <bits/stdc++.h>
#include "testlib.h"

using namespace std;

int main(int argc, char *argv[]) {
    registerValidation(argc, argv);

    int t = inf.readInt(1, 10000, "t");
    inf.readEoln();

    long long sum_n = 0;
    for (int tc = 0; tc < t; ++tc) {
        int n = inf.readInt(1, 200000, "n");
        inf.readEoln();

        string s = inf.readToken("[a-z]+", "s");
        ensuref((int)s.size() == n,
                "Length of string s (%d) does not match n=%d (test case %d)",
                (int)s.size(), n, tc + 1);
        inf.readEoln();

        sum_n += n;
        ensuref(sum_n <= 200000,
                "Sum of n over all test cases exceeds 200000 (test case %d)",
                tc + 1);
    }

    inf.readEof();
    return 0;
}