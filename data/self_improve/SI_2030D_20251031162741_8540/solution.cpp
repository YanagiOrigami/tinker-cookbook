```cpp
#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

struct SegTree {
    struct Node {
        int mx0;   // maximum value in the current state
        int mx1;   // maximum value if the whole segment is flipped
        bool lazy;
        Node(int a = -INF, int b = -INF) : mx0(a), mx1(b), lazy(false) {}
    };
    int n;
    vector<Node> st;
    vector<int> val;               // static value = prefMax[i] - i (1‑based)
    SegTree(int _n = 0) { init(_n); }
    void init(int _n) {
        n = _n;
        st.assign(4 * n + 5, Node());
        val.assign(n + 1, 0);
    }
    void build(int p, int l, int r, const vector<char>& cut) {
        if (l == r) {
            int cur = cut[l] ? val[l] : -INF;
            int flipped = cut[l] ? -INF : val[l];
            st[p] = Node(cur, flipped);
            return;
        }
        int m = (l + r) >> 1;
        build(p << 1, l, m, cut);
        build(p << 1 | 1, m + 1, r, cut);
        pull(p);
    }
    void pull(int p) {
        st[p].mx0 = max(st[p<<1].mx0, st[p<<1|1].mx0);
        st[p].mx1 = max(st[p<<1].mx1, st[p<<1|1].mx1);
    }
    void applyFlip(int p) {
        swap(st[p].mx0, st[p].mx1);
        st[p].lazy ^= 1;
    }
    void push(int p) {
        if (st[p].lazy) {
            applyFlip(p<<1);
            applyFlip(p<<1|1);
            st[p].lazy = false;
        }
    }
    // range flip [L,R]
    void rangeFlip(int p, int l, int r, int L, int R) {
        if (R < l || r < L) return;
        if (L <= l && r <= R) {
            applyFlip(p);
            return;
        }
        push(p);
        int m = (l + r) >> 1;
        rangeFlip(p<<1, l, m, L, R);
        rangeFlip(p<<1|1, m+1, r, L, R);
        pull(p);
    }
    // point set cut[i] = newCut
    void pointSet(int p, int l, int r, int idx, bool newCut) {
        if (l == r) {
            int cur = newCut ? val[l] : -INF;
            int flipped = newCut ? -INF : val[l];
            st[p] = Node(cur, flipped);
            return;
        }
        push(p);
        int m = (l + r) >> 1;
        if (idx <= m) pointSet(p<<1, l, m, idx, newCut);
        else pointSet(p<<1|1, m+1, r, idx, newCut);
        pull(p);
    }
    int queryMax() const { return st[1].mx0; }
};

struct BIT {
    int n;
    vector<int> bit;
    BIT(int _n = 0) { init(_n); }
    void init(int _n) {
        n = _n;
        bit.assign(n + 2, 0);
    }
    void add(int idx, int val) {
        for (; idx <= n; idx += idx & -idx) bit[idx] ^= (val & 1);
    }
    // range xor [l,r]
    void rangeXor(int l, int r) {
        add(l, 1);
        add(r + 1, 1);
    }
    int pointQuery(int idx) const {
        int res = 0;
        for (int i = idx; i > 0; i -= i & -i) res ^= bit[i];
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if(!(cin >> T)) return 0;
    while (T--) {
        int n, q;
        cin >> n >> q;
        vector<int> p(n + 1);
        for (int i = 1; i <= n; ++i) cin >> p[i];
        string s0;
        cin >> s0;
        s0 = " " + s0;                     // 1‑based
        // prefix maximums
        vector<int> prefMax(n + 1);
        int cur = 0;
        for (int i = 1; i <= n; ++i) {
            cur = max(cur, p[i]);
            prefMax[i] = cur;
        }
        // static values val[i] = prefMax[i] - i  (only needed for i = 1 … n‑1)
        vector<int> val(n); // size n, ignore index 0
        for (int i = 1; i < n; ++i) val[i] = prefMax[i] - i;

        // initial cut array
        vector<char> cut(n, 0); // cut[1..n-1]
        for (int i = 1; i < n; ++i) {
            if (s0[i] == 'L' && s0[i+1] == 'R') cut[i] = 1;
        }

        // segment tree over positions 1..n-1
        SegTree seg(n-1);
        seg.val = val;                     // copy static values
        seg.build(1, 1, n-1, cut);

        // BIT to store lazy range toggles of s
        BIT ft(n);
        // original s as 0/1 (0 = L, 1 = R)
        vector<int> orig(n + 1);
        for (int i = 1; i <= n; ++i) orig[i] = (s0[i] == 'R');

        auto getS = [&](int idx) -> int {
            // returns 0 for L, 1 for R after all toggles
            return orig[idx] ^ ft.pointQuery(idx);
        };

        while (q--) {
            int l, r;
            cin >> l >> r;
            // toggle s[l..r]
            ft.rangeXor(l, r);
            // internal cuts [l, r-1] are flipped
            if (l <= r-1) seg.rangeFlip(1, 1, n-1, l, r-1);
            // recompute border cuts that involve exactly one flipped character
            if (l > 1) {
                int i = l - 1;
                bool newCut = (getS(i) == 0 && getS(i+1) == 1);
                seg.pointSet(1, 1, n-1, i, newCut);
            }
            if (r < n) {
                int i = r;
                bool newCut = (getS(i) == 0 && getS(i+1) == 1);
                seg.pointSet(1, 1, n-1, i, newCut);
            }
            // answer
            cout << (seg.queryMax() <= 0 ? "YES\n" : "NO\n");
        }
    }
    return 0;
}
```