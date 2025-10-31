#include <bits/stdc++.h>
using namespace std;
using int64 = long long;
using i128 = __int128_t;

// returns total distance after d days
static inline int64 total_distance(int64 d, int64 a, int64 b, int64 c) {
    int64 cycles = d / 3;
    int64 rem = d % 3;
    int64 sum_cycle = a + b + c;
    int64 ans = cycles * sum_cycle;
    if (rem >= 1) ans += a;
    if (rem >= 2) ans += b;
    return ans;
}

// check whether day d satisfies both constraints
static bool ok(int64 d, int64 n, int64 a, int64 b, int64 c, int64 p) {
    int64 tot = total_distance(d, a, b, c);
    return tot >= n && tot % p == 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        int64 n, a, b, c, p;
        cin >> n >> a >> b >> c >> p;
        // Upper bound: we can always take d = n * max(1, p) (very loose)
        // A tighter bound uses that each day we walk at most max_abc.
        int64 max_abc = max({a, b, c});
        // In worst case we need n / min_abc days, but to be safe we use 2e18.
        int64 lo = 1, hi = (int64)4e18; // hi is exclusive
        while (lo < hi) {
            int64 mid = lo + (hi - lo) / 2;
            if (ok(mid, n, a, b, c, p))
                hi = mid;
            else
                lo = mid + 1;
        }
        cout << lo << '\n';
    }
    return 0;
}