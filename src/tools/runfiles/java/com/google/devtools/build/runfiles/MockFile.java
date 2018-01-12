// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.runfiles;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

final class MockFile implements Closeable {

  public final Path path;

  public MockFile(ImmutableList<String> lines) throws IOException {
    String testTmpdir = System.getenv("TEST_TMPDIR");
    if (Strings.isNullOrEmpty(testTmpdir)) {
      throw new IOException("$TEST_TMPDIR is empty or undefined");
    }
    path = Files.createTempFile(new File(testTmpdir).toPath(), null, null);
    Files.write(path, lines, StandardCharsets.UTF_8);
  }

  @Override
  public void close() throws IOException {
    if (path != null) {
      Files.delete(path);
    }
  }
}
