// Copyright 2014 Google Inc. All rights reserved.
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
#ifndef DEVTOOLS_BLAZE_MAIN_UTIL_STRINGS_H_
#define DEVTOOLS_BLAZE_MAIN_UTIL_STRINGS_H_

#include <string>
#include <vector>

#ifdef BLAZE_OPENSOURCE
#include <string.h>
#endif

namespace blaze_util {

using std::string;

extern const unsigned char kAsciiPropertyBits[256];
#define kApb kAsciiPropertyBits

static inline bool ascii_isspace(unsigned char c) { return kApb[c] & 0x08; }

bool starts_with(const string &haystack, const string &needle);

bool ends_with(const string &haystack, const string &needle);

// Matches a prefix (which must be a char* literal!) against the beginning of
// str. Returns a pointer past the prefix, or NULL if the prefix wasn't matched.
// (Like the standard strcasecmp(), but for efficiency doesn't call strlen() on
// prefix, and returns a pointer rather than an int.)
//
// The ""'s catch people who don't pass in a literal for "prefix"
#ifndef strprefix
#define strprefix(str, prefix) \
  (strncmp(str, prefix, sizeof("" prefix "")-1) == 0 ? \
      str + sizeof(prefix)-1 : NULL)
#endif

// Matches a prefix; returns a pointer past the prefix, or NULL if not found.
// (Like strprefix() and strcaseprefix() but not restricted to searching for
// char* literals). Templated so searching a const char* returns a const char*,
// and searching a non-const char* returns a non-const char*.
// Matches a prefix; returns a pointer past the prefix, or NULL if not found.
// (Like strprefix() and strcaseprefix() but not restricted to searching for
// char* literals). Templated so searching a const char* returns a const char*,
// and searching a non-const char* returns a non-const char*.
template<class CharStar>
inline CharStar var_strprefix(CharStar str, const char* prefix) {
  const int len = strlen(prefix);
  return strncmp(str, prefix, len) == 0 ?  str + len : NULL;
}

// Returns a mutable char* pointing to a string's internal buffer, which may not
// be null-terminated. Returns NULL for an empty string. If not non-null,
// writing through this pointer will modify the string.
inline char* string_as_array(string* str) {
  // DO NOT USE const_cast<char*>(str->data())! See the unittest for why.
  return str->empty() ? NULL : &*str->begin();
}

// Join the elements of pieces separated by delimeter.  Returns the joined
// string in output.
void JoinStrings(
    const std::vector<string> &pieces, const char delimeter, string *output);

// Splits contents by delimeter.  Skips empty subsections.
std::vector<string> Split(const string &contents, const char delimeter);

// Same as above, but adds results to output.
void SplitStringUsing(
    const string &contents, const char delimeter, std::vector<string> *output);

// Splits contents by delimeter with possible elements quoted by ' or ".
// backslashes (\) can be used to escape the quotes or delimeter. Skips
// empty subsections.
std::vector<string> SplitQuoted(const string &contents, const char delimeter);

// Same as above, but adds results to output.
void SplitQuotedStringUsing(const string &contents, const char delimeter,
                            std::vector<string> *output);

// Global replace of oldsub with newsub.
void Replace(const string &oldsub, const string &newsub, string *str);

// Removes whitespace from both ends of a string.
void StripWhitespace(string *str);

// Tokenizes str on whitespace and places the tokens in words. Splits on spaces,
// newlines, carriage returns, and tabs. Respects single and double quotes (that
// is, "a string of 'some stuff'" would be 4 tokens). If the comment character
// is found (outside of quotes), the rest of the string will be ignored. Any
// token can be escaped with \, e.g., "this\\ is\\ one\\ token".
void Tokenize(
    const string &str, const char &comment, std::vector<string> *words);

// Evaluate a format string and store the result in 'str'.
void StringPrintf(string *str, const char *format, ...);

}  // namespace blaze_util

#endif  // DEVTOOLS_BLAZE_MAIN_UTIL_STRINGS_H_
