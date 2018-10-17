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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaRuntimeInfoApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/** Information about the Java runtime used by the <code>java_*</code> rules. */
@Immutable
@AutoCodec
public class JavaRuntimeInfo extends NativeInfo implements JavaRuntimeInfoApi {
  public static final String SKYLARK_NAME = "JavaRuntimeInfo";

  public static final NativeProvider<JavaRuntimeInfo> PROVIDER =
      new NativeProvider<JavaRuntimeInfo>(JavaRuntimeInfo.class, SKYLARK_NAME) {};

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

  // Helper methods to access an instance of JavaRuntimeInfo.

  public static JavaRuntimeInfo from(RuleContext ruleContext) {
    return from(ruleContext, ":jvm", RuleConfiguredTarget.Mode.TARGET);
  }

  public static JavaRuntimeInfo forHost(RuleContext ruleContext) {
    return from(ruleContext, ":host_jdk", RuleConfiguredTarget.Mode.HOST);
  }

  public static JavaRuntimeInfo forHost(RuleContext ruleContext, String attributeSuffix) {
    return from(ruleContext, ":host_jdk" + attributeSuffix, RuleConfiguredTarget.Mode.HOST);
  }

  @Nullable
  private static JavaRuntimeInfo from(
      RuleContext ruleContext, String attributeName, RuleConfiguredTarget.Mode mode) {
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

    return collection.get(JavaRuntimeInfo.PROVIDER);
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
    super(PROVIDER);
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
  public PathFragment javaHome() {
    return javaHome;
  }

  @Override
  /** The execpath of the Java binary. */
  public PathFragment javaBinaryExecPath() {
    return javaBinaryExecPath;
  }

  /** The runfiles path of the root directory of the Java installation. */
  @Override
  public PathFragment javaHomeRunfilesPath() {
    return javaHomeRunfilesPath;
  }

  @Override
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
