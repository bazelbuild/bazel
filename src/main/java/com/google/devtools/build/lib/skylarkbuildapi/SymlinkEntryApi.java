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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Interface for a single runfiles symlink represented by a link name and target. */
@SkylarkModule(
    name = "SymlinkEntry",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "A single runfiles symlink represented by a link name and target.")
public interface SymlinkEntryApi extends StarlarkValue {

  @SkylarkCallable(
      name = "path",
      doc = "The path of the symlink in the runfiles tree",
      structField = true)
  String getPathString();

  @SkylarkCallable(name = "target_file", doc = "Target file of the symlink", structField = true)
  FileApi getArtifact();
}
