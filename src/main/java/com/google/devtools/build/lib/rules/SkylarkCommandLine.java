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

package com.google.devtools.build.lib.rules;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.BuiltinFunction;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;

/** A Skylark module class to create memory efficient command lines. */
@SkylarkModule(
  name = "cmd_helper",
  namespace = true,
  category = SkylarkModuleCategory.BUILTIN,
  doc = "Deprecated. Module for creating memory efficient command lines."
)
public class SkylarkCommandLine {

  @SkylarkSignature(
    name = "join_paths",
    objectType = SkylarkCommandLine.class,
    returnType = String.class,
    doc =
        "Deprecated. Creates a single command line argument joining the paths of a set "
            + "of files on the separator string.",
    parameters = {
      @Param(name = "separator", type = String.class, doc = "the separator string to join on."),
      @Param(
        name = "files",
        type = SkylarkNestedSet.class,
        generic1 = Artifact.class,
        doc = "the files to concatenate."
      )
    }
  )
  private static BuiltinFunction joinPaths =
      new BuiltinFunction("join_paths") {
        public String invoke(String separator, SkylarkNestedSet files) {
          NestedSet<Artifact> artifacts = files.getSet(Artifact.class);
          // TODO(bazel-team): lazy evaluate
          return Artifact.joinExecPaths(separator, artifacts);
        }
      };

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(SkylarkCommandLine.class);
  }
}
