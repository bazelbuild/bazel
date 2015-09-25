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

package com.google.devtools.build.xcode.common;

import java.nio.file.Path;

/** Defines operations common to file paths. */
public interface PathTransformer<P> {
  /** Returns the containing directory of the given path. */
  P parent(P path);

  /** Returns the result of joining a path component string to the given path. */
  P join(P path, String segment);

  /** Returns the name of the file at the given path, i.e. the last path component. */
  String name(P path);

  static final PathTransformer<Path> FOR_JAVA_PATH = new PathTransformer<Path>() {
    @Override
    public Path parent(Path path) {
      return path.getParent();
    }

    @Override
    public Path join(Path path, String segment) {
      return path.resolve(segment);
    }

    @Override
    public String name(Path path) {
      return path.getFileName().toString();
    }
  };
}
