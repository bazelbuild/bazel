// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.java;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Information about the Java runtime used by the <code>java_*</code> rules. */
@AutoValue
@Immutable
public abstract class JavaRuntimeProvider implements TransitiveInfoProvider {

  public static JavaRuntimeProvider create(
      NestedSet<Artifact> javaBaseInputs, PathFragment javaHome,
      PathFragment javaBinaryExecPath, PathFragment javaBinaryRunfilesPath) {
    return new AutoValue_JavaRuntimeProvider(
        javaBaseInputs, javaHome, javaBinaryExecPath, javaBinaryRunfilesPath);
  }

  /** All input artifacts in the javabase. */
  public abstract NestedSet<Artifact> javaBaseInputs();

  /** The root directory of the Java installation. */
  public abstract PathFragment javaHome();

  /** The execpath of the Java binary. */
  public abstract PathFragment javaBinaryExecPath();

  /** The runfiles path of the Java binary. */
  public abstract PathFragment javaBinaryRunfilesPath();
}
