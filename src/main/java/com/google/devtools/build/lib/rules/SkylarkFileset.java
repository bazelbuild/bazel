// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A wrapper class for NestedSet of Artifacts in Skylark to ensure type safety.
 */
@SkylarkModule(name = "Files", namespace = true,
    doc = "A helper class to extract path from files.")
public final class SkylarkFileset {

  @SkylarkCallable(doc = "Returns the joint execution paths of these files using the delimiter.")
  public static String joinExecPaths(String delimiter, Iterable<Artifact> artifacts) {
    return Artifact.joinExecPaths(delimiter, artifacts);
  }

  @SkylarkCallable(
      doc = "Returns a working directory for the file using suffix for the directory name")
  public static PathFragment workDir(Root root, Artifact file, String suffix) {
    PathFragment path = file.getRootRelativePath();
    String basename = FileSystemUtils.removeExtension(path.getBaseName()) + suffix;
    path = path.replaceName(basename);
    return root.getExecPath().getRelative(path);
  }
}
