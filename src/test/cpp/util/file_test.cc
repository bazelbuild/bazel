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
#include <stdio.h>
#include <string.h>

#include <memory>  // unique_ptr
#include <thread>  // NOLINT (to silence Google-internal linter)

#include "src/main/cpp/util/file.h"
#include "src/main/cpp/util/file_platform.h"
#include "gtest/gtest.h"

namespace blaze_util {

using std::string;

TEST(FileTest, TestNormalizePath) {
  ASSERT_EQ(string(""), NormalizePath(""));
  ASSERT_EQ(string(""), NormalizePath("."));
  ASSERT_EQ(string("/"), NormalizePath("/"));
  ASSERT_EQ(string("/"), NormalizePath("//"));
  ASSERT_EQ(string("foo"), NormalizePath("foo"));
  ASSERT_EQ(string("foo"), NormalizePath("foo/"));
  ASSERT_EQ(string("foo/bar"), NormalizePath("foo//bar"));
  ASSERT_EQ(string("foo/bar"), NormalizePath("../..//foo//bar"));
  ASSERT_EQ(string("/foo"), NormalizePath("/foo"));
  ASSERT_EQ(string("/foo"), NormalizePath("/foo/"));
  ASSERT_EQ(string("/foo/bar"), NormalizePath("/foo/./bar/"));
  ASSERT_EQ(string("foo/bar"), NormalizePath("../foo/baz/../bar"));
  ASSERT_EQ(string("foo/bar"), NormalizePath("../foo//./baz/../bar///"));
}

TEST(FileTest, TestSingleThreadedPipe) {
  std::unique_ptr<IPipe> pipe(CreatePipe());
  char buffer[50] = {0};
  ASSERT_TRUE(pipe.get()->Send("hello", 5));
  ASSERT_EQ(3, pipe.get()->Receive(buffer, 3));
  ASSERT_TRUE(pipe.get()->Send(" world", 6));
  ASSERT_EQ(5, pipe.get()->Receive(buffer + 3, 5));
  ASSERT_EQ(3, pipe.get()->Receive(buffer + 8, 40));
  ASSERT_EQ(0, strncmp(buffer, "hello world", 11));
}

TEST(FileTest, TestMultiThreadedPipe) {
  std::unique_ptr<IPipe> pipe(CreatePipe());
  char buffer[50] = {0};
  std::thread writer_thread([&pipe]() {
    ASSERT_TRUE(pipe.get()->Send("hello", 5));
    ASSERT_TRUE(pipe.get()->Send(" world", 6));
  });

  // Wait for all data to be fully written to the pipe.
  writer_thread.join();

  ASSERT_EQ(3, pipe.get()->Receive(buffer, 3));
  ASSERT_EQ(5, pipe.get()->Receive(buffer + 3, 5));
  ASSERT_EQ(3, pipe.get()->Receive(buffer + 8, 40));
  ASSERT_EQ(0, strncmp(buffer, "hello world", 11));
}

TEST(FileTest, TestReadFile) {
  const char* tempdir = getenv("TEST_TMPDIR");
  ASSERT_NE(nullptr, tempdir);
  ASSERT_NE(0, tempdir[0]);

  std::string filename(JoinPath(tempdir, "test.readfile"));
  FILE* fh = fopen(filename.c_str(), "wt");
  ASSERT_NE(nullptr, fh);
  ASSERT_EQ(11, fwrite("hello world", 1, 11, fh));
  fclose(fh);

  std::string actual;
  ASSERT_TRUE(ReadFile(filename, &actual));
  ASSERT_EQ(std::string("hello world"), actual);

  ASSERT_TRUE(ReadFile(filename, &actual, 5));
  ASSERT_EQ(std::string("hello"), actual);
}

}  // namespace blaze_util
