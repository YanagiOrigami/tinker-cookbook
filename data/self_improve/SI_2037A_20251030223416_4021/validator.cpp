#include <bits/stdc++.h>
#include "testlib.h"

using namespace std;

int main(int argc, char *argv[]) {
    registerValidation(argc, argv);

    int t = inf.readInt(1, 10, "t");
    inf.readEoln();

    for (int tc = 0; tc < t; ++tc) {
        int n = inf.readInt(1, 500, "n");
        inf.readEoln();

        for (int i = 0; i < n; ++i) {
            int x = inf.readInt(1, n, "a_i");
            if (i + 1 < n) {
                inf.readSpace();
            } else {
                inf.readEoln();
            }
            (void)x; // silence unused variable warning
        }
    }

    inf.readEof();
    return 0;
}