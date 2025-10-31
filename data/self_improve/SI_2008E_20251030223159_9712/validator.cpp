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

        // read the string consisting only of lowercase latin letters
        string s = inf.readToken("[a-z]+", "s");
        inf.readEoln();

        if ((int)s.size() != n) {
            quitf(_fail, "Length of string (%d) does not match n=%d (testcase %d)",
                  (int)s.size(), n, tc + 1);
        }

        sum_n += n;
    }

    ensuref(sum_n <= 200000,
            "Sum of n over all test cases is %lld, exceeds limit 200000", sum_n);

    inf.readEof();
    return 0;
}