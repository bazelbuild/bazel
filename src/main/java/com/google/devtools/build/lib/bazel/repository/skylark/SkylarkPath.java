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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkbuildapi.repository.RepositoryPathApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/**
 * A Path object to be used into Skylark remote repository.
 *
 * <p>This path object enable non-hermetic operations from Skylark and should not be returned by
 * something other than a SkylarkRepositoryContext.
 */
@Immutable
final class SkylarkPath implements RepositoryPathApi<SkylarkPath> {
  private final Path path;

  SkylarkPath(Path path) {
    this.path = path;
  }

  Path getPath() {
    return path;
  }

  @Override
  public boolean equals(Object obj) {
    return (obj instanceof SkylarkPath) &&  path.equals(((SkylarkPath) obj).path);
  }

  @Override
  public int hashCode() {
    return path.hashCode();
  }

  @Override
  public String getBasename() {
    return path.getBaseName();
  }

  @Override
  public ImmutableList<SkylarkPath> readdir() throws IOException {
    ImmutableList.Builder<SkylarkPath> builder = ImmutableList.builder();
    for (Path p : path.getDirectoryEntries()) {
      builder.add(new SkylarkPath(p));
    }
    return builder.build();
  }

  @Override
  public SkylarkPath getDirname() {
    Path parentPath = path.getParentDirectory();
    return parentPath == null ? null : new SkylarkPath(parentPath);
  }

  @Override
  public SkylarkPath getChild(String childPath) {
    return new SkylarkPath(path.getChild(childPath));
  }

  @Override
  public boolean exists() {
    return path.exists();
  }

  @Override
  public SkylarkPath realpath() throws IOException {
    return new SkylarkPath(path.resolveSymbolicLinks());
  }

  @Override
  public String toString() {
    return path.toString();
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append(toString());
  }
}
