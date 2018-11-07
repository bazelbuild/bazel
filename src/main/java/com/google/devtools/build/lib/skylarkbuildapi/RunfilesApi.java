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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * An interface for a set of runfiles.
 */
@SkylarkModule(
  name = "runfiles",
  category = SkylarkModuleCategory.NONE,
  doc = "An interface for a set of runfiles."
)
public interface RunfilesApi {

  @SkylarkCallable(
    name = "files",
    doc = "Returns the set of runfiles as files.",
    structField = true
  )
  public NestedSet<? extends FileApi> getArtifacts();

  @SkylarkCallable(name = "symlinks", doc = "Returns the set of symlinks.", structField = true)
  public NestedSet<? extends SymlinkEntryApi> getSymlinks();

  @SkylarkCallable(
      name = "root_symlinks",
      doc = "Returns the set of root symlinks.",
      structField = true)
  public NestedSet<? extends SymlinkEntryApi> getRootSymlinks();

  @SkylarkCallable(
    name = "empty_filenames",
    doc = "Returns names of empty files to create.",
    structField = true
  )
  public NestedSet<String> getEmptyFilenames();

  @SkylarkCallable(
    name = "merge",
    doc =
        "Returns a new runfiles object that includes all the contents of this one and the "
            + "argument.",
    parameters = {
        @Param(
            name = "other",
            positional = true,
            named = false,
            type = RunfilesApi.class,
            doc = "The runfiles object to merge into this."
        ),
    }
  )
  public RunfilesApi merge(RunfilesApi other);
}
