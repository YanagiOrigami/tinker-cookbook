```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<int> a(n);
    for (int &x : a) cin >> x;

    // prefZero[i] – number of zeros in prefix [0,i)
    vector<int> prefZero(n + 1, 0);
    for (int i = 0; i < n; ++i)
        prefZero[i + 1] = prefZero[i] + (a[i] == 0);

    // dp[s] – maximal passed checks after processing the current prefix
    // when exactly s points have already been spent on Strength
    // (the rest of obtained points are either spent on Intelligence or are still in the pool)
    const int INF_NEG = -1e9;
    vector<int> dp(m + 1, INF_NEG), ndp(m + 1, INF_NEG);
    dp[0] = 0;                       // no points spent yet, no checks passed

    for (int i = 0; i < n; ++i) {
        fill(ndp.begin(), ndp.end(), INF_NEG);
        int zerosHere = prefZero[i + 1] - prefZero[i]; // 1 if a[i]==0 else 0
        for (int s = 0; s <= m; ++s) if (dp[s] > INF_NEG/2) {
            int used = s;                     // points already spent on Strength
            int totalObtained = prefZero[i];  // zeros seen before position i
            int pool = totalObtained - used - ( (i>0 && a[i-1]==0) ? 0 : 0 ); // points not yet spent
            // Actually we keep pool implicitly: total spent = s + intelSpent,
            // intelSpent = (totalSpentSoFar - s). We only need to know we never spend more than pool.

            // 1) Do nothing at this record (just pass it if possible)
            int cur = dp[s];
            if (a[i] > 0) { // Intelligence check
                // We need enough points spent on Intelligence before this check.
                // intelSpent = totalSpent - s, where totalSpent ≤ totalObtained (cannot exceed pool)
                // The maximal intel we could have now is totalObtained - s (spend all remaining on Intel)
                if (totalObtained - s >= a[i]) cur = max(cur, dp[s] + 1);
            } else if (a[i] < 0) { // Strength check
                if (s >= -a[i]) cur = max(cur, dp[s] + 1);
            }
            ndp[s] = max(ndp[s], cur);

            // 2) If this record is a zero, we may spend it now on Strength
            if (a[i] == 0) {
                if (s + 1 <= m) {
                    // spend this newly obtained point on Strength
                    ndp[s + 1] = max(ndp[s + 1], dp[s]);   // spending does not give extra passed checks now
                }
                // or keep it in the pool (implicitly do nothing, already handled)
            }
        }
        dp.swap(ndp);
    }

    int ans = 0;
    for (int s = 0; s <= m; ++s) ans = max(ans, dp[s]);
    cout << ans << '\n';
    return 0;
}
```