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

import static com.google.devtools.build.lib.rules.java.JavaRuleClasses.HOST_JAVA_RUNTIME_ATTRIBUTE_NAME;
import static com.google.devtools.build.lib.rules.java.JavaRuleClasses.JAVA_RUNTIME_ATTRIBUTE_NAME;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaRuntimeInfoApi;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Information about the Java runtime used by the <code>java_*</code> rules. */
@Immutable
@AutoCodec
public final class JavaRuntimeInfo extends ToolchainInfo implements JavaRuntimeInfoApi {

  public static JavaRuntimeInfo create(
      NestedSet<Artifact> javaBaseInputs,
      NestedSet<Artifact> javaBaseInputsMiddleman,
      PathFragment javaHome,
      PathFragment javaBinaryExecPath,
      PathFragment javaHomeRunfilesPath,
      PathFragment javaBinaryRunfilesPath) {
    return new JavaRuntimeInfo(
        javaBaseInputs,
        javaBaseInputsMiddleman,
        javaHome,
        javaBinaryExecPath,
        javaHomeRunfilesPath,
        javaBinaryRunfilesPath);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  // Helper methods to access an instance of JavaRuntimeInfo.

  public static JavaRuntimeInfo forHost(RuleContext ruleContext) {
    return from(ruleContext, HOST_JAVA_RUNTIME_ATTRIBUTE_NAME, TransitionMode.HOST);
  }

  public static JavaRuntimeInfo from(RuleContext ruleContext) {
    return from(ruleContext, JAVA_RUNTIME_ATTRIBUTE_NAME, TransitionMode.TARGET);
  }

  @Nullable
  private static JavaRuntimeInfo from(
      RuleContext ruleContext, String attributeName, TransitionMode mode) {
    if (!ruleContext.attributes().has(attributeName, BuildType.LABEL)) {
      return null;
    }
    TransitiveInfoCollection prerequisite = ruleContext.getPrerequisite(attributeName, mode);
    if (prerequisite == null) {
      return null;
    }

    return from(prerequisite, ruleContext);
  }

  // TODO(katre): When all external callers are converted to use toolchain resolution, make this
  // method private.
  @Nullable
  protected static JavaRuntimeInfo from(
      TransitiveInfoCollection collection, RuleErrorConsumer errorConsumer) {
    return (JavaRuntimeInfo) collection.get(ToolchainInfo.PROVIDER);
  }

  private final NestedSet<Artifact> javaBaseInputs;
  private final NestedSet<Artifact> javaBaseInputsMiddleman;
  private final PathFragment javaHome;
  private final PathFragment javaBinaryExecPath;
  private final PathFragment javaHomeRunfilesPath;
  private final PathFragment javaBinaryRunfilesPath;

  @AutoCodec.Instantiator
  @VisibleForSerialization
  JavaRuntimeInfo(
      NestedSet<Artifact> javaBaseInputs,
      NestedSet<Artifact> javaBaseInputsMiddleman,
      PathFragment javaHome,
      PathFragment javaBinaryExecPath,
      PathFragment javaHomeRunfilesPath,
      PathFragment javaBinaryRunfilesPath) {
    super(ImmutableMap.of(), Location.BUILTIN);
    this.javaBaseInputs = javaBaseInputs;
    this.javaBaseInputsMiddleman = javaBaseInputsMiddleman;
    this.javaHome = javaHome;
    this.javaBinaryExecPath = javaBinaryExecPath;
    this.javaHomeRunfilesPath = javaHomeRunfilesPath;
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
  @Override
  public String javaHome() {
    return javaHome.toString();
  }

  public PathFragment javaHomePathFragment() {
    return javaHome;
  }

  /** The execpath of the Java binary. */
  @Override
  public String javaBinaryExecPath() {
    return javaBinaryExecPath.toString();
  }

  public PathFragment javaBinaryExecPathFragment() {
    return javaBinaryExecPath;
  }

  /** The runfiles path of the root directory of the Java installation. */
  @Override
  public String javaHomeRunfilesPath() {
    return javaHomeRunfilesPath.toString();
  }

  /** The runfiles path of the Java binary. */
  @Override
  public String javaBinaryRunfilesPath() {
    return javaBinaryRunfilesPath.toString();
  }

  public PathFragment javaBinaryRunfilesPathFragment() {
    return javaBinaryRunfilesPath;
  }

  @Override
  public Depset starlarkJavaBaseInputs() {
    return Depset.of(Artifact.TYPE, javaBaseInputs());
  }

  // Not all of JavaRuntimeInfo is exposed to Starlark, which makes implementing deep equality
  // impossible: if Java-only parts are considered, the behavior is surprising in Starlark, if they
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
