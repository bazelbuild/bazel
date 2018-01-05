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

package com.google.devtools.build.lib.rules.java;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * This class represents a Java virtual machine with a host system and a path. If the JVM comes from
 * the client, it can optionally also contain a label pointing to a target that contains all the
 * necessary files.
 */
@AutoCodec
@SkylarkModule(
  name = "jvm",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT,
  doc = "A configuration fragment representing the Java virtual machine."
)
@Immutable
public final class Jvm extends BuildConfiguration.Fragment {
  public static final ObjectCodec<Jvm> CODEC = new Jvm_AutoCodec();

  private final PathFragment javaHome;
  private final Label jvmLabel;
  private final PathFragment java;

  public static final String BIN_JAVA = "bin/java" + OsUtils.executableExtension();

  /**
   * Creates a Jvm instance. Either the {@code javaHome} parameter is absolute, and/or the {@code
   * jvmLabel} parameter must be non-null. Only the {@code jvmLabel} is optional.
   */
  @AutoCodec.Constructor
  public Jvm(PathFragment javaHome, Label jvmLabel) {
    Preconditions.checkArgument(javaHome.isAbsolute() || jvmLabel != null);
    this.javaHome = javaHome;
    this.jvmLabel = jvmLabel;
    this.java = javaHome.getRelative(BIN_JAVA);
  }

  /**
   * Returns the path to the java binary.
   *
   * <p>Don't use this method because it relies on package loading during configuration creation.
   * Use {@link JavaCommon#getHostJavaExecutable(RuleContext)} and
   * {@link JavaCommon#getJavaExecutable(RuleContext)} instead.
   */
  public PathFragment getJavaExecutable() {
    return java;
  }

  /**
   * Returns a label. Adding this label to the dependencies of an action that
   * depends on this JVM is sufficient to ensure that all the required files are
   * present. Can be <code>null</code>, in which case nothing needs to be added
   * to the dependencies of an action. We rely on convention to make sure that
   * this case works, since we can't know which JVMs are installed on the build host.
   */
  public Label getJvmLabel() {
    return jvmLabel;
  }

  public PathFragment getJavaHome() {
    return javaHome;
  }
}
