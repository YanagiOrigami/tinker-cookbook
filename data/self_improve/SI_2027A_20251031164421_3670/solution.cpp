```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        int n;
        cin >> n;
        vector<int> w(n), h(n);
        for (int i = 0; i < n; ++i) cin >> w[i] >> h[i];

        // For each stamp store a = max side, b = min side
        vector<int> A(n), B(n);
        for (int i = 0; i < n; ++i) {
            A[i] = max(w[i], h[i]); // larger side after optimal rotation
            B[i] = min(w[i], h[i]); // smaller side after optimal rotation
        }

        // pre‑compute maxA and maxB for every segment [l, r]
        const int INF = 1e9;
        vector<vector<int>> maxA(n, vector<int>(n, 0));
        vector<vector<int>> maxB(n, vector<int>(n, 0));
        for (int l = 0; l < n; ++l) {
            int curA = 0, curB = 0;
            for (int r = l; r < n; ++r) {
                curA = max(curA, A[r]);
                curB = max(curB, B[r]);
                maxA[l][r] = curA;
                maxB[l][r] = curB;
            }
        }

        // dp[i] – minimal total perimeter for the prefix of length i (0‑based)
        const long long INFLL = (1LL<<60);
        vector<long long> dp(n + 1, INFLL);
        dp[0] = 0;
        for (int i = 1; i <= n; ++i) {
            // try the last group to be [j, i-1]
            for (int j = 0; j < i; ++j) {
                int segA = maxA[j][i-1];
                int segB = maxB[j][i-1];
                long long cost = 2LL * (segA + segB);
                dp[i] = min(dp[i], dp[j] + cost);
            }
        }
        cout << dp[n] << '\n';
    }
    return 0;
}
```