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

package com.google.devtools.build.lib.skylarkbuildapi.repository;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.io.IOException;

/** A structure representing a file to be used inside a repository. */
@SkylarkModule(
    name = "path",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "A structure representing a file to be used inside a repository.")
public interface RepositoryPathApi<RepositoryPathApiT extends RepositoryPathApi<?>>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "basename",
      structField = true,
      doc = "A string giving the basename of the file.")
  String getBasename();

  @SkylarkCallable(
      name = "readdir",
      structField = false,
      doc = "The list of entries in the directory denoted by this path.")
  ImmutableList<RepositoryPathApiT> readdir() throws IOException;

  @SkylarkCallable(
      name = "dirname",
      structField = true,
      doc = "The parent directory of this file, or None if this file does not have a parent.")
  RepositoryPathApi<?> getDirname();

  @SkylarkCallable(
      name = "get_child",
      doc = "Append the given path to this path and return the resulted path.",
      parameters = {
        @Param(
            name = "child_path",
            positional = true,
            named = false,
            type = String.class,
            doc = "The path to append to this path."),
      })
  RepositoryPathApi<?> getChild(String childPath);

  @SkylarkCallable(
      name = "exists",
      structField = true,
      doc = "Returns true if the file denoted by this path exists.")
  boolean exists();

  @SkylarkCallable(
      name = "realpath",
      structField = true,
      doc =
          "Returns the canonical path for this path by repeatedly replacing all symbolic links "
              + "with their referents.")
  RepositoryPathApi<?> realpath() throws IOException;
}
