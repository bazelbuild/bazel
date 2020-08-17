#include <iostream>
#include <regex>
#include <fstream>
#include <unordered_set>

using namespace std;

string getBasename(const string &path) {
    // Assumes we're on an OS with "/" as the path separator
    auto idx = path.find_last_of("/");
    if (idx == string::npos) {
        return path;
    }
    return path.substr(idx + 1);
}

// Returns 0 if there are no duplicate basenames in the object files (both via -filelist as well as shell args),
// 1 otherwise
int main(int argc, const char * argv[]) {
    unordered_set<string> basenames;
    const regex libRegex = regex(".*\\.a$");
    const regex noArgFlags = regex("-static|-s|-a|-c|-L|-T|-no_warning_for_no_symbols");
    const regex singleArgFlags = regex("-arch_only|-syslibroot|-o");
    // Set i to 1 to skip executable path
    for (int i = 1; argv[i] != nullptr; i++) {
        string arg = argv[i];
        if (arg == "-filelist") {
            ifstream list(argv[i + 1]);
            for (string line; getline(list, line); ) {
                const string basename = getBasename(line);
                const auto pair = basenames.insert(basename);
                if (!pair.second) {
                    return 1;
                }
            }
            list.close();
            i++;
        } else if (regex_match(arg, noArgFlags)) {
            // No arg flags
        } else if (regex_match(arg, singleArgFlags)) {
            // Single-arg flags
            i++;
        } else if (arg[0] == '-') {
            return 1;
            // Unrecognized flag, let the wrapper deal with it
        } else if (regex_match(arg, libRegex)) {
            // Archive inputs can remain untouched, as they come from other targets.
        } else {
            const string basename = getBasename(arg);
            const auto pair = basenames.insert(basename);
            if (!pair.second) {
                return 1;
            }
        }
    }
    return 0;
}
