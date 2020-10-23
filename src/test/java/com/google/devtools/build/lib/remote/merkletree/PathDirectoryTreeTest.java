// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.merkletree;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.NavigableMap;
import java.util.TreeMap;

/** Tests for {@link DirectoryTreeBuilder#fromPaths}. */
public class PathDirectoryTreeTest extends DirectoryTreeTest {

  @Override
  protected DirectoryTree build(Path... paths) throws IOException {
    NavigableMap<PathFragment, Path> inputFiles = new TreeMap<>();
    for (Path path : paths) {
      inputFiles.put(path.relativeTo(execRoot), path);
    }
    return DirectoryTreeBuilder.fromPaths(inputFiles, digestUtil);
  }
}
