// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Splitter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Parse and search $PATH, the binary search path for executables.
 */
public class SearchPath {
  private static final Splitter SEPARATOR = Splitter.on(':');

  /**
   * Parses a $PATH value into a list of paths. A Null search path is treated as an empty one.
   * Relative entries in $PATH are ignored.
   */
  public static List<Path> parse(FileSystem fs, @Nullable String searchPath) {
    List<Path> paths = new ArrayList<>();
    if (searchPath == null) {
      return paths;
    }
    for (String p : SEPARATOR.split(searchPath)) {
      PathFragment pf = PathFragment.create(p);

      if (pf.isAbsolute()) {
        paths.add(fs.getPath(pf));
      }
    }
    return paths;
  }

  /**
   * Finds the first executable called {@code exe} in the searchPath.
   * If {@code exe} is not a basename, it will always return null. This should be equivalent to
   * running which(1).
   */
  @Nullable
  public static Path which(List<Path> searchPath, String exe) {
    PathFragment fragment = PathFragment.create(exe);
    if (fragment.segmentCount() != 1 || fragment.isAbsolute()) {
      return null;
    }

    for (Path p : searchPath) {
      Path ch = p.getChild(exe);

      try {
        if (ch.exists() && ch.isExecutable()) {
          return ch;
        }
      } catch (IOException e) {
        // Like which(1), we ignore any IO exception (disk on fire, permission denied etc.)
      }
    }
    return null;
  }
}
