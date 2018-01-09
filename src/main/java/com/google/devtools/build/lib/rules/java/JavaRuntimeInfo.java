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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Information about the Java runtime used by the <code>java_*</code> rules. */
@SkylarkModule(name = "JavaRuntimeInfo", doc = "Information about the Java runtime being used.")
@Immutable
public class JavaRuntimeInfo extends NativeInfo {
  public static final String SKYLARK_NAME = "JavaRuntimeInfo";

  public static final NativeProvider<JavaRuntimeInfo> PROVIDER =
      new NativeProvider<JavaRuntimeInfo>(JavaRuntimeInfo.class, SKYLARK_NAME) {};

  private final NestedSet<Artifact> javaBaseInputs;
  private final NestedSet<Artifact> javaBaseInputsMiddleman;
  private final PathFragment javaHome;
  private final PathFragment javaBinaryExecPath;
  private final PathFragment javaBinaryRunfilesPath;

  public JavaRuntimeInfo(
      NestedSet<Artifact> javaBaseInputs, NestedSet<Artifact> javaBaseInputsMiddleman,
      PathFragment javaHome, PathFragment javaBinaryExecPath,
      PathFragment javaBinaryRunfilesPath) {
    super(PROVIDER, ImmutableMap.<String, Object>of());
    this.javaBaseInputs = javaBaseInputs;
    this.javaBaseInputsMiddleman = javaBaseInputsMiddleman;
    this.javaHome = javaHome;
    this.javaBinaryExecPath = javaBinaryExecPath;
    this.javaBinaryRunfilesPath = javaBinaryRunfilesPath;
  }

  /** All input artifacts in the javabase. */
  public NestedSet<Artifact> javaBaseInputs() {
    return javaBaseInputs;
  }

  /** A middleman representing the javabase. */
  public NestedSet<Artifact> javaBaseInputsMiddleman() {
    return javaBaseInputsMiddleman;
  }

  /** The root directory of the Java installation. */
  @SkylarkCallable(
      name = "java_home",
      doc = "Returns the execpath of the root of the Java installation.",
      structField = true
  )
  public PathFragment javaHome() {
    return javaHome;
  }

  @SkylarkCallable(
      name = "java_executable_exec_path",
      doc = "Returns the execpath of the Java executable.",
      structField = true
  )
  /** The execpath of the Java binary. */
  public PathFragment javaBinaryExecPath() {
    return javaBinaryExecPath;
  }

  /** The runfiles path of the Java binary. */
  public PathFragment javaBinaryRunfilesPath() {
    return javaBinaryRunfilesPath;
  }

  // Not all of JavaRuntimeInfo is exposed to Skylark, which makes implementing deep equality
  // impossible: if Java-only parts are considered, the behavior is surprising in Skylark, if they
  // are not, the behavior is surprising in Java. Thus, object identity it is.
  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
