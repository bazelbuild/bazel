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

#if defined(_WIN32) || defined(__CYGWIN__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif  // defined(_WIN32) || defined(__CYGWIN__)

#include "src/main/cpp/util/strings.h"

#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#endif  // defined(_WIN32) || defined(__CYGWIN__)

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/main/cpp/util/exit_code.h"

namespace blaze_util {

using std::string;
using std::vector;

static const char kSeparator[] = " \n\t\r";

size_t SplitQuotedStringUsing(const string& contents, const char delimiter,
                              std::vector<string>* output) {
  size_t len = contents.length();
  size_t start = 0;
  size_t quote = string::npos;  // quote position
  size_t num_segments = 0;

  for (size_t pos = 0; pos < len; ++pos) {
    if (start == pos && contents[start] == delimiter) {
      ++start;
    } else if (contents[pos] == '\\') {
      ++pos;
    } else if (quote != string::npos && contents[pos] == contents[quote]) {
      quote = string::npos;
    } else if (quote == string::npos &&
               (contents[pos] == '"' || contents[pos] == '\'')) {
      quote = pos;
    } else if (quote == string::npos && contents[pos] == delimiter) {
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

static void GetNextToken(const string &str, const char &comment,
                         string::const_iterator *iter, vector<string> *words) {
  string output;
  auto last = *iter;
  char quote = '\0';
  // While not a delimiter.
  while (last != str.end() && (quote || strchr(kSeparator, *last) == nullptr)) {
    // Absorb escapes, except inside single quotes.
    if (*last == '\\' && quote != '\'') {
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

#if defined(_WIN32) || defined(__CYGWIN__)

template <typename U, typename V>
static bool UStrToVStr(const std::basic_string<U> &input,
                       std::basic_string<V> *output, const bool use_utf8,
                       int (*Convert)(const bool _utf8,
                                      const std::basic_string<U> &_in, V *_out,
                                      const size_t _size),
                       uint32_t *win32_error) {
  int buf_size = input.size() + 1;
  std::unique_ptr<V[]> buf(new V[buf_size]);
  // Attempt to convert, optimistically using the estimated output buffer size.
  int res = Convert(use_utf8, input, buf.get(), buf_size);
  if (res > 0) {
    *output = buf.get();
    return true;
  }

  DWORD err = GetLastError();
  if (err != ERROR_INSUFFICIENT_BUFFER) {
    if (win32_error) {
      *win32_error = static_cast<uint32_t>(err);
    }
    return false;
  }

  // The output buffer was too small. Get required buffer size.
  res = Convert(use_utf8, input, nullptr, 0);
  if (res > 0) {
    buf_size = res;
    buf.reset(new V[buf_size]);
    res = Convert(use_utf8, input, buf.get(), buf_size);
    if (res > 0) {
      *output = buf.get();
      return true;
    }
  }
  if (win32_error) {
    *win32_error = static_cast<uint32_t>(GetLastError());
  }
  return false;
}

static int ConvertWcsToMbs(const bool use_utf8, const std::wstring &input,
                           char *output, const size_t output_size) {
  return WideCharToMultiByte(use_utf8 ? CP_UTF8 : CP_ACP, 0, input.c_str(), -1,
                             output, output_size, nullptr, nullptr);
}

static int ConvertMbsToWcs(const bool /* unused */, const std::string &input,
                           wchar_t *output, const size_t output_size) {
  return MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, output,
                             output_size);
}

bool WcsToAcp(const std::wstring &input, std::string *output,
              uint32_t *win32_error) {
  return UStrToVStr(input, output, false, ConvertWcsToMbs, win32_error);
}

bool WcsToUtf8(const std::wstring &input, std::string *output,
               uint32_t *win32_error) {
  return UStrToVStr(input, output, true, ConvertWcsToMbs, win32_error);
}

bool Utf8ToWcs(const std::string &input, std::wstring *output,
               uint32_t *win32_error) {
  return UStrToVStr(input, output, /* unused */ true, ConvertMbsToWcs,
                    win32_error);
}

std::string WstringToCstring(const std::wstring &input) {
  std::string result;
  uint32_t err;
  if (!WcsToUtf8(input, &result, &err)) {
    fprintf(stderr,
            "WstringToCstring: failed with error %d (0x%08x), "
            "invalid input \"%ls\"\n",
            err, err, input.c_str());
    exit(blaze_exit_code::INTERNAL_ERROR);
  }
  return result;
}

std::wstring CstringToWstring(const std::string &input) {
  std::wstring result;
  uint32_t err;
  if (!Utf8ToWcs(input, &result, &err)) {
    fprintf(stderr,
            "CstringToWstring: failed with error %d (0x%08x), "
            "invalid input \"%s\"\n",
            err, err, input.c_str());
    exit(blaze_exit_code::INTERNAL_ERROR);
  }
  return result;
}

char** WArgsToCArgs(int argc, wchar_t **wargv) {
  char **argv = new char*[argc];
  for (int i = 0; i < argc; i++) {
    std::string arg = WstringToCstring(std::wstring(wargv[i]));
    argv[i] = new char[arg.length()+1];
    for (int j = 0; j < arg.length(); j++) {
      argv[i][j] = arg[j];
    }
    argv[i][arg.length()] = 0;
  }
  return argv;
}
#endif  // defined(_WIN32) || defined(__CYGWIN__)

}  // namespace blaze_util
