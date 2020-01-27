/* Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
   file Copyright.txt or https://cmake.org/licensing for details.  */

#ifndef BAZEL_THIRD_PARTY_DEF_PARSER_DEF_PARSER_H
#define BAZEL_THIRD_PARTY_DEF_PARSER_DEF_PARSER_H

#include <set>
#include <stdio.h>
#include <string>

std::wstring AsAbsoluteWindowsPath(const std::string& path);

class DefParser{
 public:
  DefParser() {}

  // This method adds a DEF file.
  // It merges all the symbols found in the DEF file into final result.
  bool AddDefinitionFile(const char* filename);

  // This method adds an Object file.
  // It parses that object file and merge symbols found into final result.
  bool AddObjectFile(const char* filename);

  // Add a file, the function itself will tell which type of file it is.
  bool AddFile(const std::string& filename);

  // Set the DLL name the output DEF file is used for.
  // This will cause a "LIBRARY <DLLName>" entry in the output DEF file.
  void SetDLLName(const std::string& filename);

  // Write all symbols found into the output DEF file.
  void WriteFile(FILE* file);

 private:
  std::set<std::string> Symbols;
  std::set<std::string> DataSymbols;
  std::string DLLName;

  // Returns true if filename ends with .def (case insensitive).
  static bool IsDefFile(const std::string& filename);
};

#endif
