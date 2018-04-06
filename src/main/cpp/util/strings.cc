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
#include "src/main/cpp/util/strings.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <cassert>
#include <memory>  // unique_ptr

#include "src/main/cpp/util/exit_code.h"

namespace blaze_util {

using std::string;
using std::unique_ptr;
using std::vector;
using std::wstring;

static const char kSeparator[] = " \n\t\r";

bool starts_with(const string &haystack, const string &needle) {
  return (haystack.length() >= needle.length()) &&
         (memcmp(haystack.c_str(), needle.c_str(), needle.length()) == 0);
}

template <typename char_type>
static bool ends_with_impl(const std::basic_string<char_type> &haystack,
                           const std::basic_string<char_type> &needle) {
  return (haystack.length() >= needle.length()) &&
         std::equal(haystack.cend() - needle.length(), haystack.cend(),
                    needle.cbegin());
}

bool ends_with(const string &haystack, const string &needle) {
  return ends_with_impl(haystack, needle);
}

bool ends_with(const wstring &haystack, const wstring &needle) {
  return ends_with_impl(haystack, needle);
}

void JoinStrings(const vector<string> &pieces, const char delimeter,
                 string *output) {
  bool first = true;
  for (const auto &piece : pieces) {
    if (first) {
      *output = piece;
      first = false;
    } else {
      *output += delimeter + piece;
    }
  }
}

vector<string> Split(const string &contents, const char delimeter) {
  vector<string> result;
  SplitStringUsing(contents, delimeter, &result);
  return result;
}

void SplitStringUsing(const string &contents, const char delimeter,
                      vector<string> *result) {
  assert(result);

  size_t start = 0;
  while (start < contents.length() && contents[start] == delimeter) {
    ++start;
  }

  size_t newline = contents.find(delimeter, start);
  while (newline != string::npos) {
    result->push_back(string(contents, start, newline - start));
    start = newline;
    while (start < contents.length() && contents[start] == delimeter) {
      ++start;
    }
    newline = contents.find(delimeter, start);
  }

  // If there is a trailing line, add that.
  if (start != newline && start != contents.size()) {
    result->push_back(string(contents, start));
  }
}

size_t SplitQuotedStringUsing(const string &contents, const char delimeter,
                              std::vector<string> *output) {
  size_t len = contents.length();
  size_t start = 0;
  size_t quote = string::npos;  // quote position
  size_t num_segments = 0;

  for (size_t pos = 0; pos < len; ++pos) {
    if (start == pos && contents[start] == delimeter) {
      ++start;
    } else if (contents[pos] == '\\') {
      ++pos;
    } else if (quote != string::npos && contents[pos] == contents[quote]) {
      quote = string::npos;
    } else if (quote == string::npos &&
               (contents[pos] == '"' || contents[pos] == '\'')) {
      quote = pos;
    } else if (quote == string::npos && contents[pos] == delimeter) {
      output->push_back(string(contents, start, pos - start));
      start = pos + 1;
      num_segments++;
    }
  }

  // A trailing element
  if (start < len) {
    output->push_back(string(contents, start));
    num_segments++;
  }
  return num_segments;
}

void Replace(const string &oldsub, const string &newsub, string *str) {
  size_t start = 0;
  // This is O(n^2) (the complexity of erase() is actually unspecified, but
  // usually linear).
  while ((start = str->find(oldsub, start)) != string::npos) {
    str->erase(start, oldsub.length());
    str->insert(start, newsub);
    start += newsub.length();
  }
}

void StripWhitespace(string *str) {
  int str_length = str->length();

  // Strip off leading whitespace.
  int first = 0;
  while (first < str_length && ascii_isspace(str->at(first))) {
    ++first;
  }
  // If entire string is white space.
  if (first == str_length) {
    str->clear();
    return;
  }
  if (first > 0) {
    str->erase(0, first);
    str_length -= first;
  }

  // Strip off trailing whitespace.
  int last = str_length - 1;
  while (last >= 0 && ascii_isspace(str->at(last))) {
    --last;
  }
  if (last != (str_length - 1) && last >= 0) {
    str->erase(last + 1, string::npos);
  }
}

static void GetNextToken(const string &str, const char &comment,
                         string::const_iterator *iter, vector<string> *words) {
  string output;
  auto last = *iter;
  char quote = '\0';
  // While not a delimiter.
  while (last != str.end() && (quote || strchr(kSeparator, *last) == nullptr)) {
    // Absorb escapes.
    if (*last == '\\') {
      ++last;
      if (last == str.end()) {
        break;
      }
      output += *last++;
      continue;
    }

    if (quote) {
      if (*last == quote) {
        // Absorb closing quote.
        quote = '\0';
        ++last;
      } else {
        output += *last++;
      }
    } else {
      if (*last == comment) {
        last = str.end();
        break;
      }
      if (*last == '\'' || *last == '"') {
        // Absorb opening quote.
        quote = *last++;
      } else {
        output += *last++;
      }
    }
  }

  if (!output.empty()) {
    words->push_back(output);
  }

  *iter = last;
}

void Tokenize(const string &str, const char &comment, vector<string> *words) {
  assert(words);
  words->clear();

  string::const_iterator i = str.begin();
  while (i != str.end()) {
    // Skip whitespace.
    while (i != str.end() && strchr(kSeparator, *i) != nullptr) {
      i++;
    }
    if (i != str.end() && *i == comment) {
      break;
    }
    GetNextToken(str, comment, &i, words);
  }
}

// Evaluate a format string and store the result in 'str'.
void StringPrintf(string *str, const char *format, ...) {
  assert(str);

  // Determine the required buffer size. vsnpritnf won't account for the
  // terminating '\0'.
  va_list args;
  va_start(args, format);
  int output_size = vsnprintf(nullptr, 0, format, args);
  if (output_size < 0) {
    fprintf(stderr, "Fatal error formatting string: %d", output_size);
    exit(blaze_exit_code::INTERNAL_ERROR);
  }
  va_end(args);

  // Allocate a buffer and format the input.
  int buffer_size = output_size + sizeof '\0';
  char *buf = new char[buffer_size];
  va_start(args, format);
  int print_result = vsnprintf(buf, buffer_size, format, args);
  if (print_result < 0) {
    fprintf(stderr, "Fatal error formatting string: %d", print_result);
    exit(blaze_exit_code::INTERNAL_ERROR);
  }
  va_end(args);

  *str = buf;
  delete[] buf;
}

void ToLower(string *str) {
  assert(str);
  *str = AsLower(*str);
}

string AsLower(const string &str) {
  if (str.empty()) {
    return "";
  }
  unique_ptr<char[]> result(new char[str.size() + 1]);
  char *result_ptr = result.get();
  for (const auto &ch : str) {
    *result_ptr++ = tolower(ch);
  }
  result.get()[str.size()] = 0;
  return string(result.get());
}

template <typename U, typename V>
static unique_ptr<V[]> UstringToVstring(
    const U *input, size_t (*convert)(V *output, const U *input, size_t len),
    const char *fmtStringU) {
  size_t size = convert(nullptr, input, 0) + 1;
  if (size == (size_t)-1) {
    fprintf(stderr, "UstringToVstring: invalid input \"");
    fprintf(stderr, fmtStringU, input);
    fprintf(stderr, "\"\n");
    exit(blaze_exit_code::INTERNAL_ERROR);
    return unique_ptr<V[]>(nullptr);  // formally return, though unreachable
  }
  unique_ptr<V[]> result(new V[size]);
  convert(result.get(), input, size);
  result.get()[size - 1] = 0;
  return std::move(result);
}

unique_ptr<char[]> WstringToCstring(const wchar_t *input) {
  return UstringToVstring<wchar_t, char>(input, wcstombs, "%ls");
}

std::string WstringToString(const std::wstring &input) {
  return string(WstringToCstring(input.c_str()).get());
}

unique_ptr<wchar_t[]> CstringToWstring(const char *input) {
  return UstringToVstring<char, wchar_t>(input, mbstowcs, "%s");
}

}  // namespace blaze_util
