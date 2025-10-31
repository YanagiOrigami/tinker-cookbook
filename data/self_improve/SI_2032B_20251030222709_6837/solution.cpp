```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T; 
    if(!(cin >> T)) return 0;
    while (T--) {
        int n; long long k;
        cin >> n >> k;
        vector<long long> a(n + 1);
        for (int i = 1; i <= n; ++i) cin >> a[i];

        // transform values:
        //  >k : +1 , <k : -1 , =k : 0
        vector<int> pref(n + 1, 0);
        for (int i = 1; i <= n; ++i) {
            int add = (a[i] > k) - (a[i] < k);
            pref[i] = pref[i - 1] + add;
        }

        // first position of k
        int posk = -1;
        for (int i = 1; i <= n; ++i) if (a[i] == k) { posk = i; break; }
        if (posk == -1) {                 // no k at all
            cout << -1 << '\n';
            continue;
        }

        // For each parity (0 even, 1 odd) we store the earliest index
        // where a given prefix sum appeared.
        const int SHIFT = n + 5;          // to make indices non‑negative
        const int MAXV = 2 * n + 10;
        vector< array<int,2> > first(MAXV, array<int,2>{-1,-1});

        // initialise with prefix before the first element of the answer,
        // i.e. pref[posk-1] at its parity.
        int init_val = pref[posk - 1] + SHIFT;
        first[init_val][ (posk - 1) & 1 ] = posk - 1;

        bool found = false;
        int L = -1, R = -1;

        for (int r = posk; r <= n && !found; ++r) {
            int cur = pref[r] + SHIFT;
            int parity = r & 1;
            // we need earlier index with same prefix sum and same parity
            if (first[cur][parity] != -1) {
                int l = first[cur][parity] + 1; // subarray [l … r]
                // it automatically contains position posk because first index ≤ posk-1
                L = l; R = r;
                found = true;
                break;
            }
            // store the first occurrence of (pref[r], parity) if not stored yet
            if (first[cur][parity] == -1) first[cur][parity] = r;
        }

        if (!found) cout << -1 << '\n';
        else cout << L << ' ' << R << '\n';
    }
    return 0;
}
```