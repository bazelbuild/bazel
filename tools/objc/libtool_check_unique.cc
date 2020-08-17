#include <iostream>
#include <regex>
#include <fstream>
#include <unordered_set>

using namespace std;

static unordered_set<string> sBasenames;

string getBasename(string path) {
    auto idx = path.find_last_of("/");
    if (idx == string::npos) {
        return path;
    }
    return path.substr(idx + 1);
}

// Returns 0 if there are no duplicate basenames in the object files (both via -filelist as well as shell args),
// 1 otherwise
int main(int argc, const char * argv[]) {
    regex no_arg_flags = regex("-static|-s|-a|-c|-L|-T|-no_warning_for_no_symbols");
    regex single_arg_flags = regex("-arch_only|-syslibroot|-o");
    // Set i to 1 to skip executable path
    for (int i = 1; argv[i] != nullptr; i++) {
        string arg = argv[i];
        if (arg == "-filelist") {
            ifstream list(argv[i + 1]);
            for (string line; getline(list, line); ) {
                string basename = getBasename(line);
                auto pair = sBasenames.insert(basename);
                if (!pair.second) {
                    return 1;
                }
            }
            list.close();
            i++;
        } else if (regex_match(arg, no_arg_flags)) {
            // No arg flags
        } else if (regex_match(arg, single_arg_flags)) {
            // Single-arg flags
            i++;
        } else if (arg[0] == '-') {
            return 1;
            // Unrecognized flag, let the wrapper deal with it
        } else if (regex_match(arg, regex(".*\\.a$"))) {
            // Archive inputs can remain untouched, as they come from other targets.
        } else {
            string basename = getBasename(arg);
            auto pair = sBasenames.insert(basename);
            if (!pair.second) {
                return 1;
            }
        }
    }
    return 0;
}
