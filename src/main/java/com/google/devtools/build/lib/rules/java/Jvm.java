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

import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * This class represents a Java virtual machine with a host system and a path.
 * If the JVM comes from the client, it can optionally also contain a label
 * pointing to a target that contains all the necessary files.
 */
@SkylarkModule(name = "jvm",
    doc = "A configuration fragment representing the Java virtual machine.")
@Immutable
public final class Jvm extends BuildConfiguration.Fragment {
  private final PathFragment javaHome;
  private final Label jvmLabel;
  private final PathFragment javac;
  private final PathFragment jar;
  private final PathFragment java;

  /**
   * Creates a Jvm instance. Either the {@code javaHome} parameter is absolute,
   * or the {@code jvmLabel} parameter must be non-null. This restriction might
   * be lifted in the future. Only the {@code jvmLabel} is optional.
   */
  public Jvm(PathFragment javaHome, Label jvmLabel) {
    Preconditions.checkArgument(javaHome.isAbsolute() ^ (jvmLabel != null));
    this.javaHome = javaHome;
    this.jvmLabel = jvmLabel;
    this.javac = getJavaHome().getRelative("bin/javac" + OsUtils.executableExtension());
    this.jar = getJavaHome().getRelative("bin/jar" + OsUtils.executableExtension());
    this.java = getJavaHome().getRelative("bin/java" + OsUtils.executableExtension());
  }

  @Override
  public void addImplicitLabels(Multimap<String, Label> implicitLabels) {
    if (jvmLabel != null) {
      implicitLabels.put("Jvm", jvmLabel);
    }
  }

  /**
   * Returns a path fragment that determines the path to the installation
   * directory. It is either absolute or relative to the execution root.
   */
  public PathFragment getJavaHome() {
    return javaHome;
  }

  /**
   * Returns the path to the javac binary.
   */
  public PathFragment getJavacExecutable() {
    return javac;
  }

  /**
   * Returns the path to the jar binary.
   */
  public PathFragment getJarExecutable() {
    return jar;
  }

  /**
   * Returns the path to the java binary.
   */
  @SkylarkCallable(name = "java_executable", structField = true,
      doc = "The java executable, i.e. bin/java relative to the Java home.")
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

  @Override
  public void addGlobalMakeVariables(Builder<String, String> globalMakeEnvBuilder) {
    globalMakeEnvBuilder.put("JAVABASE", getJavaHome().getPathString());
    globalMakeEnvBuilder.put("JAVA", getJavaExecutable().getPathString());
  }
}
