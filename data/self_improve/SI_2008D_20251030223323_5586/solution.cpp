#include <bits/stdc++.h>
using namespace std;

struct CycleInfo {
    int len;                     // length of the cycle
    long long totalBlack;        // number of black vertices in the whole cycle
    vector<int> pref;            // prefix sums of black values, size len+1
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        int n;
        cin >> n;
        vector<int> p(n + 1);
        for (int i = 1; i <= n; ++i) cin >> p[i];
        string s;
        cin >> s;
        int q;
        cin >> q;
        
        // data for each vertex
        vector<int> cycId(n + 1, -1);
        vector<int> posInCyc(n + 1, -1);
        vector<CycleInfo> cycles;
        vector<char> used(n + 1, 0);
        
        for (int v = 1; v <= n; ++v) if (!used[v]) {
            // discover one whole cycle
            vector<int> verts;
            int cur = v;
            while (!used[cur]) {
                used[cur] = 1;
                verts.push_back(cur);
                cur = p[cur];
            }
            int L = (int)verts.size();
            CycleInfo ci;
            ci.len = L;
            ci.pref.assign(L + 1, 0);
            for (int i = 0; i < L; ++i) {
                int node = verts[i];
                int isBlack = (s[node - 1] == '0');
                ci.pref[i + 1] = ci.pref[i] + isBlack;
                // store vertex info
                cycId[node] = (int)cycles.size();
                posInCyc[node] = i;
            }
            ci.totalBlack = ci.pref[L];
            cycles.push_back(std::move(ci));
        }
        
        while (q--) {
            int v;
            long long k;
            cin >> v >> k;
            int id = cycId[v];
            const CycleInfo &ci = cycles[id];
            int L = ci.len;
            long long steps = k + 1;               // number of visited vertices
            long long full = steps / L;
            int rem = (int)(steps % L);
            long long ans = full * ci.totalBlack;
            if (rem) {
                int start = posInCyc[v];
                int endPos = start + rem;
                if (endPos <= L) {
                    ans += ci.pref[endPos] - ci.pref[start];
                } else {
                    // wrapped around the end of the cycle
                    ans += ci.totalBlack - (ci.pref[start] - ci.pref[endPos - L]);
                }
            }
            cout << ans << '\n';
        }
    }
    return 0;
}