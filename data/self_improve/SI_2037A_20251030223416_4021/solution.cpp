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
        vector<int> a(n + 1);
        for (int i = 1; i <= n; ++i) cin >> a[i];

        // positions[value] = list of indices where this value appears
        vector<vector<int>> pos(n + 1);
        for (int i = 1; i <= n; ++i) pos[a[i]].push_back(i);

        // dp[l][r] = maximum number of pairs in subarray [l, r]
        vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));

        for (int len = 2; len <= n; ++len) {
            for (int l = 1; l + len - 1 <= n; ++l) {
                int r = l + len - 1;
                // option: do not use position r
                dp[l][r] = dp[l][r - 1];

                // try to pair r with some i (l <= i < r) having the same value
                const vector<int> &vec = pos[a[r]];
                // iterate over indices i in vec that are < r
                for (int idx = (int)vec.size() - 1; idx >= 0; --idx) {
                    int i = vec[idx];
                    if (i < l) break;          // all earlier i are even smaller
                    if (i >= r) continue;      // skip i == r (cannot happen)
                    int cur = dp[l][i - 1] + 1 + dp[i + 1][r - 1];
                    if (cur > dp[l][r]) dp[l][r] = cur;
                }
            }
        }
        cout << dp[1][n] << '\n';
    }
    return 0;
}