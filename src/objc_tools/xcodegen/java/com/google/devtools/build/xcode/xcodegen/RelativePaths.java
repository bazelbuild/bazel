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

package com.google.devtools.build.xcode.xcodegen;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.ImmutableList;

import java.nio.file.FileSystem;
import java.nio.file.Path;
import java.util.List;

/**
 * Utility methods for working with relative {@link Path} objects.
 */
class RelativePaths {
  private RelativePaths() {
    throw new UnsupportedOperationException("static-only");
  }

  /**
   * Converts the given string to a {@code Path}, confirming it is relative.
   */
  static Path fromString(FileSystem fileSystem, String pathString) {
    Path path = fileSystem.getPath(pathString);
    checkArgument(!path.isAbsolute(), "Expected relative path but got: %s", path);
    return path;
  }

  /**
   * Converts each item in {@code pathStrings} using {@link #fromString(FileSystem, String)}.
   */
  static List<Path> fromStrings(FileSystem fileSystem, Iterable<String> pathStrings) {
    ImmutableList.Builder<Path> result = new ImmutableList.Builder<>();
    for (String pathString : pathStrings) {
      result.add(fromString(fileSystem, pathString));
    }
    return result.build();
  }
}
