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

#include <memory>  // unique_ptr
#include <string>
#include <vector>

#ifdef BLAZE_OPENSOURCE
#include <string.h>
#endif

namespace blaze_util {

// Space characters according to Python: chr(i).isspace()
static inline bool ascii_isspace(unsigned char c) {
  return c == 9       // TAB
         || c == 10   // LF
         || c == 11   // VT (vertical tab)
         || c == 12   // FF (form feed)
         || c == 13   // CR
         || c == 32;  // space
}

bool starts_with(const std::string &haystack, const std::string &needle);

bool ends_with(const std::string &haystack, const std::string &needle);

bool ends_with(const std::wstring &haystack, const std::wstring &needle);

// Matches a prefix (which must be a char* literal!) against the beginning of
// str. Returns a pointer past the prefix, or NULL if the prefix wasn't matched.
// (Like the standard strcasecmp(), but for efficiency doesn't call strlen() on
// prefix, and returns a pointer rather than an int.)
//
// The ""'s catch people who don't pass in a literal for "prefix"
#ifndef strprefix
#define strprefix(str, prefix)                         \
  (strncmp(str, prefix, sizeof("" prefix "") - 1) == 0 \
       ? str + sizeof(prefix) - 1                      \
       : NULL)
#endif

// Matches a prefix; returns a pointer past the prefix, or NULL if not found.
// (Like strprefix() and strcaseprefix() but not restricted to searching for
// char* literals). Templated so searching a const char* returns a const char*,
// and searching a non-const char* returns a non-const char*.
// Matches a prefix; returns a pointer past the prefix, or NULL if not found.
// (Like strprefix() and strcaseprefix() but not restricted to searching for
// char* literals). Templated so searching a const char* returns a const char*,
// and searching a non-const char* returns a non-const char*.
template <class CharStar>
inline CharStar var_strprefix(CharStar str, const char *prefix) {
  const int len = strlen(prefix);
  return strncmp(str, prefix, len) == 0 ? str + len : NULL;
}

// Join the elements of pieces separated by delimeter.  Returns the joined
// string in output.
void JoinStrings(const std::vector<std::string> &pieces, const char delimeter,
                 std::string *output);

// Splits contents by delimeter.  Skips empty subsections.
std::vector<std::string> Split(const std::string &contents,
                               const char delimeter);

// Same as above, but adds results to output.
void SplitStringUsing(const std::string &contents, const char delimeter,
                      std::vector<std::string> *output);

// Same as above, but adds results to output. Returns number of elements added.
size_t SplitQuotedStringUsing(const std::string &contents, const char delimeter,
                              std::vector<std::string> *output);

// Global replace of oldsub with newsub.
void Replace(const std::string &oldsub, const std::string &newsub,
             std::string *str);

// Removes whitespace from both ends of a string.
void StripWhitespace(std::string *str);

// Tokenizes str on whitespace and places the tokens in words. Splits on spaces,
// newlines, carriage returns, and tabs. Respects single and double quotes (that
// is, "a string of 'some stuff'" would be 4 tokens). If the comment character
// is found (outside of quotes), the rest of the string will be ignored. Any
// token can be escaped with \, e.g., "this\\ is\\ one\\ token".
void Tokenize(const std::string &str, const char &comment,
              std::vector<std::string> *words);

// Evaluate a format string and store the result in 'str'.
void StringPrintf(std::string *str, const char *format, ...);

// Convert str to lower case. No locale handling, this is just for ASCII.
void ToLower(std::string *str);

std::string AsLower(const std::string &str);

// Convert a wchar_t string to a char string. Useful when consuming results of
// widechar Windows API functions.
// TODO(laszlocsomor): audit usages of WstringToCstring and replace with
// WcsToAcp or WcsToUtf8 appropriately. WstringToCstring does not specify the
// output encoding.
//
// Deprecated. Use WcsToAcp or WcsToUtf8.
std::unique_ptr<char[]> WstringToCstring(const wchar_t *input);

// Deprecated. Use WcsToAcp or WcsToUtf8.
std::string WstringToString(const std::wstring &input);

// Convert a char string to a wchar_t string. Useful when passing arguments to
// widechar Windows API functions.
// TODO(laszlocsomor): audit usages of CstringToWstring and replace with
// AcpToWcs or Utf8ToWcs appropriately. CstringToWstring does not specify the
// input encoding.
//
// Deprecated. Use AcpToWcs or Utf8ToWcs.
std::unique_ptr<wchar_t[]> CstringToWstring(const char *input);

// Deprecated. Use AcpToWcs or Utf8ToWcs.
std::wstring CstringToWstring(const std::string &input);

#if defined(_WIN32) || defined(__CYGWIN__)
// Convert UTF-16 string to ASCII (using the Active Code Page).
bool WcsToAcp(const std::wstring &input, std::string *output,
              uint32_t *error = nullptr);

// Convert UTF-16 string to UTF-8.
bool WcsToUtf8(const std::wstring &input, std::string *output,
               uint32_t *error = nullptr);

// Convert ASCII string (using the Active Code Page) to UTF-16 string.
bool AcpToWcs(const std::string &input, std::wstring *output,
              uint32_t *error = nullptr);

// Convert UTF-8 string to UTF-16.
bool Utf8ToWcs(const std::string &input, std::wstring *output,
               uint32_t *error = nullptr);
#endif  // defined(_WIN32) || defined(__CYGWIN__)

}  // namespace blaze_util

#endif  // BAZEL_SRC_MAIN_CPP_UTIL_STRINGS_H_
