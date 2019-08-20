// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.junctions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.windows.jni.WindowsFileOperations;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Junction creator implementation for Windows.
 *
 * <p>Creates a junction (or uses a cached one) for a path. If the path is a directory, the junction
 * points to it, and the returned path is the junction's path. If the path is a file, the junction
 * points to its parent, and the returned path is the file's path through the junction.
 *
 * <p>The `close` method deletes all junctions that this object created, along with the `dir`
 * directory where the junctions are created. The purpose of this is to avoid other methods (such as
 * ScopedTemporaryDirectory.close) to traverse these junctions believing they are regular
 * directories and deleting files in them that are actually outside of the directory tree.
 */
public final class WindowsJunctionCreator implements JunctionCreator {
  private final Path dir;
  private Map<Path, Path> paths; // allocated lazily, but semantically final
  private int junctionIndex = 0;

  public WindowsJunctionCreator(Path dir) {
    this.dir = Preconditions.checkNotNull(dir);
  }

  @Nullable
  public Path create(@Nullable Path path) throws IOException {
    if (path == null) {
      return null;
    }

    if (paths == null) {
      paths = new LinkedHashMap<>();
    }
    path = path.toAbsolutePath();
    if (path.toFile().isDirectory()) {
      Path link = paths.get(path);
      if (link == null) {
        link = dir.resolve(Integer.toString(junctionIndex++));
        WindowsFileOperations.createJunction(link.toString(), path.toString());
        paths.put(path, link);
      }
      return link;
    }

    Path parent = path.getParent();
    return (parent == null) ? path : create(parent).resolve(path.getFileName());
  }

  @Override
  public void close() throws IOException {
    // Delete all junctions, otherwise the temp directory deleter would follow them and delete files
    // from directories they point to.
    if (paths != null) {
      for (Path link : paths.values()) {
        link.toFile().delete();
      }
    }
    dir.toFile().delete();
  }
}
