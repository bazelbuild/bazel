// Copyright 2016 The Bazel Authors. All rights reserved.
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

#ifndef THIRD_PARTY_BAZEL_SRC_TOOLS_SINGLEJAR_TOKEN_STREAM_H_
#define THIRD_PARTY_BAZEL_SRC_TOOLS_SINGLEJAR_TOKEN_STREAM_H_ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "src/main/cpp/util/path_platform.h"
#include "src/tools/singlejar/diag.h"

/*
 * Tokenize command line containing indirect command line arguments.
 * An '@' at the beginning of a command line argument indicates that
 * the rest of the argument is the name of the file which should be
 * read and tokenized as Bash does it: tokens are separated by the
 * whitespace, quotes and double quotes can be used to have whitespace
 * and the other quote inside the token, and backslash followed by
 * newline is treated as empty string.
 */

class ArgTokenStream {
  /* This class is used as follows:
   *
   *  int main(int argc, char* argv[]) {
   *    ArgTokenStream tokens(argc-1, argv+1);
   *    while (!tokens.AtEnd()) {
   *       if (tokens.MatchAndSet("--opt1", ...) ||
   *           tokens.MatchAndSet("--opt2", ...) ||
   *           ...) {
   *         continue;
   *       }
   *       // Process non-option argument or report an error.
   *       // ArgTokenStream::token() returns the current token.
   *    }
   *  }
   */

 private:
  // Internal class to handle indirect command files.
  class FileTokenStream {
   public:
    FileTokenStream(const char *filename) {
#ifdef _WIN32
      std::wstring wpath;
      std::string error;
      if (!blaze_util::AsAbsoluteWindowsPath(filename, &wpath, &error)) {
        diag_err(1, "%s:%d: AsAbsoluteWindowsPath failed: %s", __FILE__,
                 __LINE__, error.c_str());
      }
      fp_ = _wfopen(wpath.c_str(), L"r");
#else
      fp_ = fopen(filename, "r");
#endif

      if (!fp_) {
        diag_err(1, "%s", filename);
      }
      filename_ = filename;
      next_char();
    }

    ~FileTokenStream() { close(); }

    // Assign next token to TOKEN, return true on success, false on EOF.
    bool next_token(std::string *token) {
      if (!fp_) {
        return false;
      }
      *token = "";
      while (current_char_ != EOF && isspace(current_char_)) {
        next_char();
      }
      if (current_char_ == EOF) {
        close();
        return false;
      }
      for (;;) {
        if (current_char_ == '\'' || current_char_ == '"') {
          process_quoted(token);
          if (isspace(current_char_)) {
            next_char();
            return true;
          } else {
            next_char();
          }
        } else if (current_char_ == '\\') {
          next_char();
          if ((current_char_ != EOF)) {
            token->push_back(current_char_);
            next_char();
          } else {
            diag_errx(1, "Expected character after \\, got EOF in %s",
                      filename_.c_str());
          }
        } else if (current_char_ == EOF || isspace(current_char_)) {
          next_char();
          return true;
        } else {
          token->push_back(current_char_);
          next_char();
        }
      }
    }

   private:
    void close() {
      if (fp_) {
        fclose(fp_);
        fp_ = nullptr;
      }
      filename_.clear();
    }

    // Append the quoted string to the TOKEN. The quote character (which can be
    // single or double quote) is in the current character. Everything up to the
    // matching quote character is appended.
    void process_quoted(std::string *token) {
      char quote = current_char_;
      next_char();
      while (current_char_ != quote) {
        if (current_char_ == '\\' && quote == '"') {
          // In the "-quoted token, \" stands for ", and \x
          // is copied literally for any other x.
          next_char();
          if (current_char_ == '"') {
            token->push_back('"');
            next_char();
          } else if (current_char_ != EOF) {
            token->push_back('\\');
            token->push_back(current_char_);
            next_char();
          } else {
            diag_errx(1, "No closing %c in %s", quote, filename_.c_str());
          }
        } else if (current_char_ != EOF) {
          token->push_back(current_char_);
          next_char();
        } else {
          diag_errx(1, "No closing %c in %s", quote, filename_.c_str());
        }
      }
    }

    // Get the next character from the input stream. Skip backslash followed
    // by the newline.
    void next_char() {
      if (feof(fp_)) {
        current_char_ = EOF;
        return;
      }
      current_char_ = getc(fp_);
      // Eat "\\\n" sequence.
      while (current_char_ == '\\') {
        int c = getc(fp_);
        if (c == '\n') {
          current_char_ = getc(fp_);
        } else {
          if (c != EOF) {
            ungetc(c, fp_);
          }
          break;
        }
      }
    }

    FILE *fp_;
    std::string filename_;
    int current_char_;
  };

 public:
  // Constructor. Automatically reads the first token.
  ArgTokenStream(int argc, const char *const *argv)
      : argv_(argv), argv_end_(argv + argc) {
    next();
  }

  // Process --OPTION
  // If the current token is --OPTION, set given FLAG to true, proceed to next
  // token and return true
  bool MatchAndSet(const char *option, bool *flag) {
    if (token_.compare(option) != 0) {
      return false;
    }
    *flag = true;
    next();
    return true;
  }

  // Process --OPTION OPTARG
  // If the current token is --OPTION, set OPTARG to the next token, proceed to
  // the next token after it and return true.
  bool MatchAndSet(const char *option, std::string *optarg) {
    if (token_.compare(option) != 0) {
      return false;
    }
    next();
    if (AtEnd()) {
      diag_errx(1, "%s requires argument", option);
    }
    *optarg = token_;
    next();
    return true;
  }

  // Process --OPTION OPTARG1 OPTARG2 ...
  // If a current token is --OPTION, push_back all subsequent tokens up to the
  // next option to the OPTARGS array, proceed to the next option and return
  // true.
  bool MatchAndSet(const char *option, std::vector<std::string> *optargs) {
    if (token_.compare(option) != 0) {
      return false;
    }
    next();
    while (!AtEnd() && '-' != token_.at(0)) {
      optargs->push_back(token_);
      next();
    }
    return true;
  }

  // Process --OPTION OPTARG1 OPTARG2 ...
  // If a current token is --OPTION, insert all subsequent tokens up to the
  // next option to the OPTARGS set, proceed to the next option and return
  // true.
  bool MatchAndSet(const char *option, std::set<std::string> *optargs) {
    if (token_ != option) {
      return false;
    }
    next();
    while (!AtEnd() && '-' != token_.at(0)) {
      optargs->insert(token_);
      next();
    }
    return true;
  }

  // Process --OPTION OPTARG1,OPTSUFF1 OPTARG2,OPTSUFF2 ...
  // If a current token is --OPTION, push_back all subsequent tokens up to the
  // next option to the OPTARGS array, splitting the OPTARG,OPTSUFF by a comma,
  // proceed to the next option and return true.
  bool MatchAndSet(const char *option,
                   std::vector<std::pair<std::string, std::string> > *optargs) {
    if (token_.compare(option) != 0) {
      return false;
    }
    next();
    while (!AtEnd() && '-' != token_.at(0)) {
      size_t commapos = token_.find(',');
      if (commapos == std::string::npos) {
        optargs->push_back(std::pair<std::string, std::string>(token_, ""));
      } else {
        std::string first = token_.substr(0, commapos);
        token_.erase(0, commapos + 1);
        optargs->push_back(std::pair<std::string, std::string>(first, token_));
      }

      next();
    }
    return true;
  }

  // Current token.
  const std::string &token() const { return token_; }

  // Read the next token.
  void next() {
    if (AtEnd()) {
      return;
    }
    if (file_token_stream_.get() && token_from_file()) {
      return;
    }
    while (argv_ < argv_end_) {
      if (**argv_ != '@') {
        token_ = *argv_++;
        return;
      }
      file_token_stream_.reset(new FileTokenStream(*(argv_++) + 1));
      if (token_from_file()) {
        return;
      }
    }
    argv_++;
  }

  // True if there are no more tokens.
  bool AtEnd() const { return argv_ > argv_end_; }

 private:
  bool token_from_file() {
    if (file_token_stream_->next_token(&token_)) {
      return true;
    }
    file_token_stream_.reset(nullptr);
    return false;
  }
  std::unique_ptr<FileTokenStream> file_token_stream_;
  const char *const *argv_;
  const char *const *argv_end_;
  std::string token_;
};

#endif  //  THIRD_PARTY_BAZEL_SRC_TOOLS_SINGLEJAR_TOKEN_STREAM_H_
