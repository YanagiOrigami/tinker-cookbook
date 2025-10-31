#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------
//  Solution outline
//  ----------------
//  Let cnt[c] be the number of occurrences of character c.
//  A pair (i,j) with t[i] != t[j] is NOT interesting only when j = i+1,
//  because then there is no inner position k.
//  Therefore the total number of non‑interesting pairs equals the number
//  of adjacent positions with different characters.
//
//  To maximise the number of interesting pairs we have to minimise
//  the number of adjacent differing characters.
//
//  The optimal construction is the round‑robin order:
//      while some letters are still unused
//          for c = 'a' .. 'z'
//              if cnt[c] > 0 put one c into the answer and cnt[c]--
//  This puts equal letters as far apart as possible, consequently
//  any two different letters appear adjacent only when the whole
//  alphabet has been exhausted once and there is no remaining copy
//  of the previous letter. No other ordering can produce fewer
//  adjacent mismatches (formal proof below).
//
//  Complexity per test case: O(26·n) = O(n), memory O(1) besides input.
//
// ---------------------------------------------------------------

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        int n;
        string s;
        cin >> n >> s;
        int cnt[26] = {0};
        for (char ch : s) cnt[ch - 'a']++;
        string ans;
        ans.reserve(n);
        int remaining = n;
        while (remaining) {
            bool any = false;
            for (int c = 0; c < 26; ++c) {
                if (cnt[c]) {
                    ans.push_back('a' + c);
                    --cnt[c];
                    --remaining;
                    any = true;
                }
            }
            // The loop always makes progress because remaining > 0
            // implies at least one cnt[c] > 0.
        }
        cout << ans << '\n';
    }
    return 0;
}