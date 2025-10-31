```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        long long a[2], b[2], s[2];
        for (int i = 0; i < 2; ++i) cin >> a[i] >> b[i] >> s[i];

        long long L = 2, R = 1000000000LL;          // current possible interval for x

        for (int i = 0; i < 2; ++i) {
            long long delta = s[i] - a[i] * b[i];
            if (delta == 0) {
                // x > max(a,b)
                L = max(L, max(a[i], b[i]) + 1);
            } else if (delta == a[i] + b[i] + 1) {
                // x <= min(a,b)
                R = min(R, min(a[i], b[i]));
            } else if (delta == a[i]) {
                // only b >= x, thus a < x <= b
                L = max(L, a[i] + 1);
                R = min(R, b[i]);
            } else if (delta == b[i]) {
                // only a >= x, thus b < x <= a
                L = max(L, b[i] + 1);
                R = min(R, a[i]);
            } else {
                // impossible according to statement, but keep safety
            }
        }
        // After two queries the interval is guaranteed to contain exactly one value.
        cout << L << '\n';
    }
    return 0;
}
```