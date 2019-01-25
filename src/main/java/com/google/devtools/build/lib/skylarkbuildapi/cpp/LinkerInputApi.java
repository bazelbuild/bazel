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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * Something that appears on the command line of the linker. Since we sometimes expand archive files
 * to their constituent object files, we need to keep information whether a certain file contains
 * embedded objects and if so, the list of the object files themselves.
 */
@SkylarkModule(
    name = "LinkerInputApi",
    category = SkylarkModuleCategory.BUILTIN,
    documented = false,
    doc = "An input that appears in the command line of the linker.")
public interface LinkerInputApi<FileT extends FileApi> {
  /** Returns the artifact that is the input of the linker. */
  @SkylarkCallable(name = "artifact", doc = "Artifact passed to the linker.")
  FileT getArtifact();

  @SkylarkCallable(name = "original_artifact", doc = "Artifact passed to the linker.")
  FileT getOriginalLibraryArtifact();
}
