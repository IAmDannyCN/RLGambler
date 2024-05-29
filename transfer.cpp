#include <bits/stdc++.h>
//#include <windows.h>

#define bug cout<<"bug "<<__LINE__<<endl
#define index xedni
//#define int long long

const int inf=1e9+7;
const int MOD=1e9+7;
//const int MAXN=;

using namespace std;

signed main() {
	freopen("score_table.txt", "r", stdin);
	freopen("score_dict.txt", "w", stdout);
    ios_base::sync_with_stdio(false);
    cin.tie(0); cout.tie(0);

    cout << "{";
    int rk, ty, sum=0;
    while(cin >> std::oct >> rk >> std::dec >> ty) {
        if (rk == -1) {
            sum += ty;
        } else {
            cout << ty << " : " << 2250 - rk - sum << ", " ;
        }
    } cout << "}" << endl;

    return 0;
}