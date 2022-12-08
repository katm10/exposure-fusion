#include "utils.h"

using std::string;
using std::map;
using std::cout;
using std::cerr;
using std::endl;

using namespace Halide;

// This applied a compute_root() schedule to all the Func's that are consumed by
// the calling Func
void apply_auto_schedule(Func F) {
    map<string,Internal::Function> flist = Internal::find_transitive_calls(F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    map<string,Internal::Function>::iterator fit;
    for (fit=flist.begin(); fit!=flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        cout << "Warning: applying default schedule to " << f.name() << endl;
    }
    cout << endl;
}
