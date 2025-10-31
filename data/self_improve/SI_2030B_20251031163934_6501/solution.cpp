```cpp
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
using namespace std;
using boost::multiprecision::cpp_int;

// compute C(k, L) exactly, but stop as soon as the value exceeds limit
cpp_int binom_limit(int k, int L, const cpp_int& limit) {
    if (L > k) return 0;
    cpp_int res = 1;
    for (int i = 1; i <= L; ++i) {
        res *= (k - L + i);
        res /= i;
        if (res > limit) return limit + 1; // early stop
    }
    return res;
}

// compute exact C(k, L) (k <= 2e5, L <= k)
cpp_int binom_exact(int k, int L) {
    if (L > k) return 0;
    cpp_int res = 1;
    for (int i = 1; i <= L; ++i) {
        res *= (k - L + i);
        res /= i;
    }
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T; 
    if(!(cin >> T)) return 0;
    while (T--) {
        int n, L;
        cin >> n >> L;
        // total number of subsequences of length L
        cpp_int total = binom_exact(n, L);
        cpp_int half = total / 2;                 // floor(total/2)

        // binary search the smallest k (L <= k <= n) with C(k,L) >= half
        int low = L, high = n, best = n;
        while (low <= high) {
            int mid = (low + high) >> 1;
            cpp_int cur = binom_limit(mid, L, half);
            if (cur >= half) {
                best = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }

        // candidate ks are best and best-1 (if possible)
        int opt_k = L;
        cpp_int bestDiff = total;  // huge initial value

        auto evaluate = [&](int k) {
            if (k < L) return;
            cpp_int c0 = binom_exact(k, L);
            cpp_int diff = (c0 * 2 > total) ? (c0 * 2 - total) : (total - c0 * 2);
            if (diff < bestDiff) {
                bestDiff = diff;
                opt_k = k;
            }
        };

        evaluate(best);
        evaluate(best - 1);
        evaluate(best + 1); // safety, may be out of range

        // construct any string with opt_k zeros and (n - opt_k) ones
        string ans;
        ans.append(opt_k, '0');
        ans.append(n - opt_k, '1');
        cout << ans << '\n';
    }
    return 0;
}
```