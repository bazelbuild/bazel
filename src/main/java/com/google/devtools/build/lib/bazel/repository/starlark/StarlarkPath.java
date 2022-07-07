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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkValue;

/**
 * A Path object to be used into Starlark remote repository.
 *
 * <p>This path object enable non-hermetic operations from Starlark and should not be returned by
 * something other than a StarlarkRepositoryContext.
 */
@Immutable
@StarlarkBuiltin(
    name = "path",
    category = DocCategory.BUILTIN,
    doc = "A structure representing a file to be used inside a repository.")
final class StarlarkPath implements StarlarkValue {
  private final Path path;

  StarlarkPath(Path path) {
    this.path = path;
  }

  Path getPath() {
    return path;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public boolean equals(Object obj) {
    return (obj instanceof StarlarkPath) && path.equals(((StarlarkPath) obj).path);
  }

  @Override
  public int hashCode() {
    return path.hashCode();
  }

  @StarlarkMethod(
      name = "basename",
      structField = true,
      doc = "A string giving the basename of the file.")
  public String getBasename() {
    return path.getBaseName();
  }

  @StarlarkMethod(
      name = "readdir",
      structField = false,
      doc = "The list of entries in the directory denoted by this path.")
  public ImmutableList<StarlarkPath> readdir() throws IOException {
    ImmutableList.Builder<StarlarkPath> builder = ImmutableList.builder();
    for (Path p : path.getDirectoryEntries()) {
      builder.add(new StarlarkPath(p));
    }
    return builder.build();
  }

  @StarlarkMethod(
      name = "dirname",
      structField = true,
      allowReturnNones = true,
      doc = "The parent directory of this file, or None if this file does not have a parent.")
  @Nullable
  public StarlarkPath getDirname() {
    Path parentPath = path.getParentDirectory();
    return parentPath == null ? null : new StarlarkPath(parentPath);
  }

  @StarlarkMethod(
      name = "get_child",
      doc = "Append the given path to this path and return the resulted path.",
      parameters = {
        @Param(name = "child_path", doc = "The path to append to this path."),
      })
  public StarlarkPath getChild(String childPath) {
    return new StarlarkPath(path.getChild(childPath));
  }

  @StarlarkMethod(
      name = "exists",
      structField = true,
      doc = "Returns true if the file denoted by this path exists.")
  public boolean exists() {
    return path.exists();
  }

  @StarlarkMethod(
      name = "realpath",
      structField = true,
      doc =
          "Returns the canonical path for this path by repeatedly replacing all symbolic links "
              + "with their referents.")
  public StarlarkPath realpath() throws IOException {
    return new StarlarkPath(path.resolveSymbolicLinks());
  }

  @Override
  public String toString() {
    return path.toString();
  }

  @Override
  public void repr(Printer printer) {
    printer.append(toString());
  }
}
