// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef BAZEL_SRC_MAIN_CPP_UTIL_STRINGS_H_
#define BAZEL_SRC_MAIN_CPP_UTIL_STRINGS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace blaze_util {

// Matches a prefix; returns a pointer past the prefix, or nullptr if not found.
// (Like strprefix() and strcaseprefix() but not restricted to searching for
// char* literals). Templated so searching a const char* returns a const char*,
// and searching a non-const char* returns a non-const char*.
template <class CharStar>
inline CharStar var_strprefix(CharStar str, const char *prefix) {
  const int len = strlen(prefix);
  return strncmp(str, prefix, len) == 0 ? str + len : nullptr;
}

// Splits contents by delimiter, skipping empty subsections, while respecting
// single/double quotes and backslash escapes. Adds results to output and
// returns number of elements added.
size_t SplitQuotedStringUsing(const std::string& contents, char delimiter,
                              std::vector<std::string>* output);

// Tokenizes str on whitespace and places the tokens in words. Splits on spaces,
// newlines, carriage returns, and tabs. Respects single and double quotes (that
// is, "a string of 'some stuff'" would be 4 tokens). If the comment character
// is found (outside of quotes), the rest of the string will be ignored. Any
// token can be escaped with \, e.g., "this\\ is\\ one\\ token".
void Tokenize(const std::string &str, const char &comment,
              std::vector<std::string> *words);

#if defined(_WIN32) || defined(__CYGWIN__)
// Convert UTF-16 string to ASCII (using the Active Code Page).
bool WcsToAcp(const std::wstring &input, std::string *output,
              uint32_t *error = nullptr);

// Convert UTF-16 string to UTF-8.
bool WcsToUtf8(const std::wstring &input, std::string *output,
               uint32_t *error = nullptr);

// Convert UTF-8 string to UTF-16.
bool Utf8ToWcs(const std::string &input, std::wstring *output,
               uint32_t *error = nullptr);

// Deprecated. Use WcsToAcp or WcsToUtf8.
std::string WstringToCstring(const std::wstring &input);

// Deprecated. Use AcpToWcs or Utf8ToWcs.
std::wstring CstringToWstring(const std::string &input);

char** WArgsToCArgs(int argc, wchar_t **wargv);
#endif  // defined(_WIN32) || defined(__CYGWIN__)

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_STRINGS_H_
