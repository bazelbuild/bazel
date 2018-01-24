/* Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
   file Copyright.txt or https://cmake.org/licensing for details.  */

#include <fstream>
#include <iostream>
#include <string>

#include "third_party/def_parser/def_parser.h"

static const char* ws = " \t\n\r\f\v";

inline void trim(std::string *str) {
  // Remove prefixing spaces
  str->erase(0, str->find_first_not_of(ws));
  // Remove suffixing spaces
  str->erase(str->find_last_not_of(ws) + 1);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: output_def_file dllname [objfile ...] [input_deffile ...] [@paramfile ...]\n";
    std::cerr << "output_deffile: the output DEF file\n";
    std::cerr << "\n";
    std::cerr << "dllname: the DLL name this DEF file is used for, if dllname is not empty\n";
    std::cerr << "         string (eg. ""), def_parser writes an 'LIBRARY <dllname>' entry\n";
    std::cerr << "         into DEF file.\n";
    std::cerr << "\n";
    std::cerr << "objfile: a object file, def_parser parses this file to find symbols,\n";
    std::cerr << "         then merges them into final result.\n";
    std::cerr << "         Can apppear multiple times.\n";
    std::cerr << "\n";
    std::cerr << "input_deffile: an existing def file, def_parser merges all symbols in this file.\n";
    std::cerr << "               Can appear multiple times.\n";
    std::cerr << "\n";
    std::cerr << "@paramfile: a parameter file that can contain objfile and input_deffile.\n";
    std::cerr << "            Can appear multiple time.\n";
    return 1;
  }

  std::wstring filenameW = AsAbsoluteWindowsPath(argv[1]);
  FILE* fout = _wfopen(filenameW.c_str(), L"w+");
  if (!fout) {
    std::cerr << "Could not open output .def file: " << argv[1] << "\n";
    return 1;
  }

  DefParser deffile;

  deffile.SetDLLName(argv[2]);

  for (int i = 3; i < argc; i++) {
    // If the argument starts with @, then treat it as a parameter file.
    if (argv[i][0] == '@') {
      filenameW = AsAbsoluteWindowsPath(argv[i] + 1);
      std::ifstream paramfile(filenameW.c_str(), std::ios::in | std::ios::binary);
      if (!paramfile) {
        std::cerr << "Could not open parameter file: " << argv[i] << "\n";
        return 1;
      }
      std::string file;
      while (std::getline(paramfile, file)) {
        trim(&file);
        if (!deffile.AddFile(file)) {
          return 1;
        }
      }
    } else {
      std::string file(argv[i]);
      trim(&file);
      if (!deffile.AddFile(argv[i])) {
        return 1;
      }
    }
  }

  deffile.WriteFile(fout);
  fclose(fout);
  return 0;
}
