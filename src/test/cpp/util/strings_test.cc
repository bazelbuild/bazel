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

#include <wchar.h>

#include <memory>
#include <string>
#include <vector>

#include "googletest/include/gtest/gtest.h"

namespace blaze_util {

using std::string;
using std::unique_ptr;
using std::vector;

TEST(BlazeUtil, JoinStrings) {
  vector<string> pieces;
  string output;
  JoinStrings(pieces, ' ', &output);
  ASSERT_EQ("", output);

  pieces.push_back("abc");
  JoinStrings(pieces, ' ', &output);
  ASSERT_EQ("abc", output);

  pieces.push_back("");
  JoinStrings(pieces, ' ', &output);
  ASSERT_EQ("abc ", output);

  pieces.push_back("def");
  JoinStrings(pieces, ' ', &output);
  ASSERT_EQ("abc  def", output);
}

TEST(BlazeUtil, Split) {
  string lines = "";
  vector<string> pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(0), pieces.size());

  lines = "foo";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "\nfoo";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "\n\n\nfoo";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "foo\n";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "foo\n\n\n";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "foo\nbar";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(2), pieces.size());
  ASSERT_EQ("foo", pieces[0]);
  ASSERT_EQ("bar", pieces[1]);

  lines = "foo\n\nbar";
  pieces = Split(lines, '\n');
  ASSERT_EQ(size_t(2), pieces.size());
  ASSERT_EQ("foo", pieces[0]);
  ASSERT_EQ("bar", pieces[1]);
}

TEST(BlazeUtil, Replace) {
  string line = "foo\\\nbar\nbaz";
  Replace("\\\n", "", &line);
  ASSERT_EQ("foobar\nbaz", line);

  line = "foo\\\n\\\nbar";
  Replace("\\\n", "", &line);
  ASSERT_EQ("foobar", line);

  line = "foo\\\r\nbar";
  Replace("\\\r\n", "", &line);
  ASSERT_EQ("foobar", line);

  line = "\\\n\\\r\n";
  Replace("\\\n", "", &line);
  Replace("\\\r\n", "", &line);
  ASSERT_EQ("", line);

  line = "x:y:z";
  Replace(":", "_C", &line);
  ASSERT_EQ("x_Cy_Cz", line);

  line = "x_::y_:__z";
  Replace("_", "_U", &line);
  Replace(":", "_C", &line);
  ASSERT_EQ("x_U_C_Cy_U_C_U_Uz", line);
}

TEST(BlazeUtil, StripWhitespace) {
  string str = "   ";
  StripWhitespace(&str);
  ASSERT_EQ("", str);

  str = "    abc  ";
  StripWhitespace(&str);
  ASSERT_EQ("abc", str);

  str = "abc";
  StripWhitespace(&str);
  ASSERT_EQ("abc", str);
}

TEST(BlazeUtil, Tokenize) {
  vector<string> result;
  string str = "a b c";
  Tokenize(str, '#', &result);
  ASSERT_EQ(size_t(3), result.size());
  EXPECT_EQ("a", result[0]);
  EXPECT_EQ("b", result[1]);
  EXPECT_EQ("c", result[2]);

  str = "a 'b c'";
  Tokenize(str, '#', &result);
  ASSERT_EQ(size_t(2), result.size());
  EXPECT_EQ("a", result[0]);
  EXPECT_EQ("b c", result[1]);

  str = "foo# bar baz";
  Tokenize(str, '#', &result);
  ASSERT_EQ(size_t(1), result.size());
  EXPECT_EQ("foo", result[0]);

  str = "foo # bar baz";
  Tokenize(str, '#', &result);
  ASSERT_EQ(size_t(1), result.size());
  EXPECT_EQ("foo", result[0]);

  str = "#bar baz";
  Tokenize(str, '#', &result);
  ASSERT_EQ(size_t(0), result.size());

  str = "#";
  Tokenize(str, '#', &result);
  ASSERT_EQ(size_t(0), result.size());

  str = " \tfirst second /    ";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(3), result.size());
  EXPECT_EQ("first", result[0]);
  EXPECT_EQ("second", result[1]);
  EXPECT_EQ("/", result[2]);

  str = " \tfirst second /    ";
  Tokenize(str, '/', &result);
  ASSERT_EQ(size_t(2), result.size());
  EXPECT_EQ("first", result[0]);
  EXPECT_EQ("second", result[1]);

  str = "first \"second' third\" fourth";
  Tokenize(str, '/', &result);
  ASSERT_EQ(size_t(3), result.size());
  EXPECT_EQ("first", result[0]);
  EXPECT_EQ("second' third", result[1]);
  EXPECT_EQ("fourth", result[2]);

  str = "first 'second\" third' fourth";
  Tokenize(str, '/', &result);
  ASSERT_EQ(size_t(3), result.size());
  EXPECT_EQ("first", result[0]);
  EXPECT_EQ("second\" third", result[1]);
  EXPECT_EQ("fourth", result[2]);

  str = "\\ this\\ is\\ one\\'\\ token";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(1), result.size());
  EXPECT_EQ(" this is one' token", result[0]);

  str = "\\ this\\ is\\ one\\'\\ token\\";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(1), result.size());
  EXPECT_EQ(" this is one' token", result[0]);

  str = "unterminated \" runs to end of line";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(2), result.size());
  EXPECT_EQ("unterminated", result[0]);
  EXPECT_EQ(" runs to end of line", result[1]);

  str = "";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(0), result.size());

  str = "one two\'s three";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(2), result.size());
  EXPECT_EQ("one", result[0]);
  EXPECT_EQ("twos three", result[1]);

  str = "one \'two three";
  Tokenize(str, 0, &result);
  ASSERT_EQ(size_t(2), result.size());
  EXPECT_EQ("one", result[0]);
  EXPECT_EQ("two three", result[1]);
}

static vector<string> SplitQuoted(const string &contents,
                                  const char delimeter) {
  vector<string> result;
  SplitQuotedStringUsing(contents, delimeter, &result);
  return result;
}

TEST(BlazeUtil, SplitQuoted) {
  string lines = "";
  vector<string> pieces = SplitQuoted(lines, '\n');
  ASSERT_EQ(size_t(0), pieces.size());

  // Same behaviour without quotes as Split
  lines = "foo";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = " foo";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "   foo";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "foo ";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "foo   ";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo", pieces[0]);

  lines = "foo bar";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(2), pieces.size());
  ASSERT_EQ("foo", pieces[0]);
  ASSERT_EQ("bar", pieces[1]);

  lines = "foo  bar";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(2), pieces.size());
  ASSERT_EQ("foo", pieces[0]);
  ASSERT_EQ("bar", pieces[1]);

  // Test with quotes
  lines = "' 'foo";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("' 'foo", pieces[0]);

  lines = " ' ' foo";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(2), pieces.size());
  ASSERT_EQ("' '", pieces[0]);
  ASSERT_EQ("foo", pieces[1]);

  lines = "foo' \\' ' ";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo' \\' '", pieces[0]);

  lines = "foo'\\'\" ' ";
  pieces = SplitQuoted(lines, ' ');
  ASSERT_EQ(size_t(1), pieces.size());
  ASSERT_EQ("foo'\\'\" '", pieces[0]);
}

TEST(BlazeUtil, StringPrintf) {
  string out;
  StringPrintf(&out, "%s %s", "a", "b");
  EXPECT_EQ("a b", out);
}

TEST(BlazeUtil, CstringToWstringTest) {
  unique_ptr<wchar_t[]> actual = CstringToWstring("hello world");
  EXPECT_EQ(0, wcscmp(actual.get(), L"hello world"));
}

TEST(BlazeUtil, WstringToCstringTest) {
  unique_ptr<char[]> actual = WstringToCstring(L"hello world");
  EXPECT_EQ(0, strcmp(actual.get(), "hello world"));
}

TEST(BlazeUtil, EndsWithTest) {
  ASSERT_TRUE(ends_with("", ""));
  ASSERT_TRUE(ends_with(L"", L""));

  ASSERT_TRUE(ends_with("abc", "bc"));
  ASSERT_TRUE(ends_with(L"abc", L"bc"));

  // prefix matches but suffix doesn't
  ASSERT_FALSE(ends_with("abc", "bd"));
  ASSERT_FALSE(ends_with(L"abc", L"bd"));

  // suffix matches but prefix doesn't
  ASSERT_FALSE(ends_with("abc", "dc"));
  ASSERT_FALSE(ends_with(L"abc", L"dc"));

  // full match
  ASSERT_TRUE(ends_with("abc", "abc"));
  ASSERT_TRUE(ends_with(L"abc", L"abc"));

  // bigger "needle" than "haystack"
  ASSERT_FALSE(ends_with("bc", "abc"));
  ASSERT_FALSE(ends_with(L"bc", L"abc"));

  ASSERT_FALSE(ends_with("bc", "def"));
  ASSERT_FALSE(ends_with(L"bc", L"def"));
}

}  // namespace blaze_util
