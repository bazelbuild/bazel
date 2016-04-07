/* Copyright (c) 2015, Google Inc.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
 * OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
 * CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#ifndef OPENSSL_HEADER_CRYPTO_TEST_FILE_TEST_H
#define OPENSSL_HEADER_CRYPTO_TEST_FILE_TEST_H

#include <stdint.h>
#include <stdio.h>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4702)
#endif

#include <string>
#include <map>
#include <set>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// File-based test framework.
//
// This module provides a file-based test framework. The file format is based on
// that of OpenSSL upstream's evp_test and BoringSSL's aead_test. Each input
// file is a sequence of attributes, blocks, and blank lines.
//
// Each attribute has the form:
//
//   Name = Value
//
// Either '=' or ':' may be used to delimit the name from the value. Both the
// name and value have leading and trailing spaces stripped.
//
// Blocks are delimited by lines beginning with three hyphens, "---". One such
// line begins a block and another ends it. Blocks are intended as a convenient
// way to embed PEM data and include their delimiters.
//
// Outside a block, lines beginning with # are ignored.
//
// A test is a sequence of one or more attributes followed by a block or blank
// line. Blank lines are otherwise ignored. For tests that process multiple
// kinds of test cases, the first attribute is parsed out as the test's type and
// parameter. Otherwise, attributes are unordered. The first attribute is also
// included in the set of attributes, so tests which do not dispatch may ignore
// this mechanism.
//
// Functions in this module freely output to |stderr| on failure. Tests should
// also do so, and it is recommended they include the corresponding test's line
// number in any output. |PrintLine| does this automatically.
//
// Each attribute in a test must be consumed. When a test completes, if any
// attributes haven't been processed, the framework reports an error.


class FileTest {
 public:
  explicit FileTest(const char *path);
  ~FileTest();

  // is_open returns true if the file was successfully opened.
  bool is_open() const { return file_ != nullptr; }

  enum ReadResult {
    kReadSuccess,
    kReadEOF,
    kReadError,
  };

  // ReadNext reads the next test from the file. It returns |kReadSuccess| if
  // successfully reading a test and |kReadEOF| at the end of the file. On
  // error or if the previous test had unconsumed attributes, it returns
  // |kReadError|.
  ReadResult ReadNext();

  // PrintLine is a variant of printf which prepends the line number and appends
  // a trailing newline.
  void PrintLine(const char *format, ...)
#ifdef __GNUC__
      __attribute__((__format__(__printf__, 2, 3)))
#endif
  ;

  unsigned start_line() const { return start_line_; }

  // GetType returns the name of the first attribute of the current test.
  const std::string &GetType();
  // GetParameter returns the value of the first attribute of the current test.
  const std::string &GetParameter();
  // GetBlock returns the optional block of the current test, or the empty
  // if there was no block.
  const std::string &GetBlock();

  // HasAttribute returns true if the current test has an attribute named |key|.
  bool HasAttribute(const std::string &key);

  // GetAttribute looks up the attribute with key |key|. It sets |*out_value| to
  // the value and returns true if it exists and returns false with an error to
  // |stderr| otherwise.
  bool GetAttribute(std::string *out_value, const std::string &key);

  // GetAttributeOrDie looks up the attribute with key |key| and aborts if it is
  // missing. It only be used after a |HasAttribute| call.
  const std::string &GetAttributeOrDie(const std::string &key);

  // GetBytes looks up the attribute with key |key| and decodes it as a byte
  // string. On success, it writes the result to |*out| and returns
  // true. Otherwise it returns false with an error to |stderr|. The value may
  // be either a hexadecimal string or a quoted ASCII string. It returns true on
  // success and returns false with an error to |stderr| on failure.
  bool GetBytes(std::vector<uint8_t> *out, const std::string &key);

  // ExpectBytesEqual returns true if |expected| and |actual| are equal.
  // Otherwise, it returns false and prints a message to |stderr|.
  bool ExpectBytesEqual(const uint8_t *expected, size_t expected_len,
                        const uint8_t *actual, size_t actual_len);

 private:
  void ClearTest();
  void OnKeyUsed(const std::string &key);

  FILE *file_ = nullptr;
  // line_ is the number of lines read.
  unsigned line_ = 0;

  // start_line_ is the line number of the first attribute of the test.
  unsigned start_line_ = 0;
  // type_ is the name of the first attribute of the test.
  std::string type_;
  // parameter_ is the value of the first attribute.
  std::string parameter_;
  // attributes_ contains all attributes in the test, including the first.
  std::map<std::string, std::string> attributes_;
  // block_, if non-empty, is the test's optional trailing block.
  std::string block_;

  // unused_attributes_ is the set of attributes that have been queried.
  std::set<std::string> unused_attributes_;
  // used_block_ is true if the block has been queried.
  bool used_block_ = false;

  FileTest(const FileTest&) = delete;
  FileTest &operator=(const FileTest&) = delete;
};

// FileTestMain runs a file-based test out of |path| and returns an exit code
// suitable to return out of |main|. |run_test| should return true on pass and
// false on failure. FileTestMain also implements common handling of the 'Error'
// attribute. A test with that attribute is expected to fail. The value of the
// attribute is the reason string of the expected OpenSSL error code.
//
// Tests are guaranteed to run serially and may affect global state if need be.
// It is legal to use "tests" which, for example, import a private key into a
// list of keys. This may be used to initialize a shared set of keys for many
// tests. However, if one test fails, the framework will continue to run
// subsequent tests.
int FileTestMain(bool (*run_test)(FileTest *t, void *arg), void *arg,
                 const char *path);


#endif /* OPENSSL_HEADER_CRYPTO_TEST_FILE_TEST_H */
