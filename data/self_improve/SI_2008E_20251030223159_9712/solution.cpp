#include <bits/stdc++.h>
using namespace std;

/*
   Alternating String Reconstruction
   -------------------------------------------------
   For each pair of letters (x, y) we keep two DP arrays:
   even[x][y] – length of the longest subsequence built so far
                whose length is even (i.e. we are ready to take
                a character equal to x next).
   odd[x][y]  – length of the longest subsequence whose length is odd
                (next expected letter is y).

   While scanning the original string we only extend a subsequence
   when the current character matches the expected letter,
   because keeping a mismatching character cannot increase the number
   of matches and therefore never helps to minimise the total cost.
   Final answer = n - max_even, where max_even is the largest even[x][y].

   Complexity:   O(26 * n) per test case   ( ≤ 5.2·10⁶ operations )
   Memory   :    26 * 26 * 2 integers ≈ 1352 integers.
*/

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        int n;
        string s;
        cin >> n >> s;
        // dp arrays
        static int even[26][26];
        static int odd[26][26];
        // initialise with zeros
        for (int i = 0; i < 26; ++i)
            for (int j = 0; j < 26; ++j)
                even[i][j] = odd[i][j] = 0;

        for (char ch : s) {
            int c = ch - 'a';
            // current character can serve as the even‑position letter (x)
            for (int y = 0; y < 26; ++y) {
                // extend an even‑length subsequence to odd length
                // taking this character (it matches x)
                odd[c][y] = max(odd[c][y], even[c][y] + 1);
            }
            // current character can serve as the odd‑position letter (y)
            for (int x = 0; x < 26; ++x) {
                // extend an odd‑length subsequence to even length
                // taking this character (it matches y)
                even[x][c] = max(even[x][c], odd[x][c] + 1);
            }
        }

        int best = 0;
        for (int x = 0; x < 26; ++x)
            for (int y = 0; y < 26; ++y)
                best = max(best, even[x][y]);   // only even lengths are valid

        cout << (n - best) << '\n';
    }
    return 0;
}