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

package com.google.devtools.build.lib.bazel.repository.skylark;

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.vfs.Path;

/**
 * A Path object to be used into Skylark remote repository.
 */
@Immutable
@SkylarkModule(name = "path", doc = "A structure representing a file to be used inside a repository"
)
final class SkylarkPath {
  final Path path;

  public SkylarkPath(Path path) {
    this.path = path;
  }

  @SkylarkCallable(
    name = "basename",
    structField = true,
    doc = "A string giving the basename of the file."
  )
  public String getBasename() {
    return path.getBaseName();
  }

  @SkylarkCallable(
    name = "dirname",
    structField = true,
    doc = "The parent directory of this file, or None if this file does not have a parent."
  )
  public SkylarkPath getDirname() {
    Path parentPath = path.getParentDirectory();
    return parentPath == null ? null : new SkylarkPath(parentPath);
  }

  @SkylarkCallable(
    name = "get_child",
    doc = "Append the given path to this path and return the resulted path."
  )
  public SkylarkPath getChild(String childPath) {
    return new SkylarkPath(path.getChild(childPath));
  }

  @SkylarkCallable(
    name = "exists",
    structField = true,
    doc = "Returns true if the file denoted by this path exists."
  )
  public boolean exists() {
    return path.exists();
  }

  @Override
  public String toString() {
    return path.toString();
  }
}
