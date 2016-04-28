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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A helper class for compiling Java targets. This helper does not rely on the
 * presence of rule-specific attributes.
 */
public class BaseJavaCompilationHelper {
  protected final RuleContext ruleContext;
  private final String implicitAttributesSuffix;
  protected final JavaToolchainProvider javaToolchain;

  public BaseJavaCompilationHelper(RuleContext ruleContext) {
    this(ruleContext, "");
  }

  public BaseJavaCompilationHelper(RuleContext ruleContext, String implicitAttributesSuffix) {
    this.ruleContext = ruleContext;
    this.implicitAttributesSuffix = implicitAttributesSuffix;
    this.javaToolchain =
        ruleContext.getPrerequisite(
            ":java_toolchain" + implicitAttributesSuffix, Mode.TARGET, JavaToolchainProvider.class);
  }

  /**
   * Returns the artifacts required to invoke {@code javahome} relative binary
   * in the action.
   */
  public NestedSet<Artifact> getHostJavabaseInputsNonStatic(RuleContext ruleContext) {
    // This must have a different name than above, because the middleman creation uses the rule's
    // configuration, although it should use the host configuration.
    return AnalysisUtils.getMiddlemanFor(ruleContext, ":host_jdk" + implicitAttributesSuffix);
  }

  /**
   * Returns the instrumentation jar in the given semantics.
   */
  protected Iterable<Artifact> getInstrumentationJars() {
    TransitiveInfoCollection instrumentationTarget = ruleContext.getPrerequisite(
        "$jacoco_instrumentation" + implicitAttributesSuffix, Mode.HOST);
    if (instrumentationTarget == null) {
      return ImmutableList.<Artifact>of();
    }
    return FileType.filter(
        instrumentationTarget.getProvider(FileProvider.class).getFilesToBuild(),
        JavaSemantics.JAR);
  }

  /**
   * Returns the javac bootclasspath artifacts from the given toolchain (if it has any) or the rule.
   */
  public static ImmutableList<Artifact> getBootClasspath(
      RuleContext ruleContext,
      JavaToolchainProvider javaToolchain,
      String implicitAttributesSuffix) {
    return ImmutableList.copyOf(javaToolchain.getBootclasspath());
  }

  /**
   * Returns the javac bootclasspath artifacts.
   */
  protected final ImmutableList<Artifact> getBootClasspath() {
    return getBootClasspath(ruleContext, javaToolchain, implicitAttributesSuffix);
  }

  /**
   * Returns the extdir artifacts.
   */
  protected final ImmutableList<Artifact> getExtdirInputs() {
    return ImmutableList.copyOf(javaToolchain.getExtclasspath());
  }

  private Artifact getIjarArtifact(Artifact jar, boolean addPrefix) {
    if (addPrefix) {
      PathFragment ruleBase = ruleContext.getUniqueDirectory("_ijar");
      PathFragment artifactDirFragment = jar.getRootRelativePath().getParentDirectory();
      String ijarBasename = FileSystemUtils.removeExtension(jar.getFilename()) + "-ijar.jar";
      return ruleContext.getDerivedArtifact(
          ruleBase.getRelative(artifactDirFragment).getRelative(ijarBasename),
          ruleContext.getConfiguration().getGenfilesDirectory());
    } else {
      return derivedArtifact(jar, "", "-ijar.jar");
    }
  }

  /**
   * Creates the Action that creates ijars from Jar files.
   *
   * @param inputJar the Jar to create the ijar for
   * @param addPrefix whether to prefix the path of the generated ijar with the package and
   *     name of the current rule
   * @return the Artifact to create with the Action
   */
  protected Artifact createIjarAction(Artifact inputJar, boolean addPrefix) {
    Artifact interfaceJar = getIjarArtifact(inputJar, addPrefix);
    FilesToRunProvider ijarTarget = javaToolchain.getIjar();
    if (!ruleContext.hasErrors()) {
      ruleContext.registerAction(new SpawnAction.Builder()
          .addInput(inputJar)
          .addOutput(interfaceJar)
          .setExecutable(ijarTarget)
          // On Windows, ijar.exe needs msys-2.0.dll and zlib1.dll in PATH.
          // Use default shell environment so that those can be found.
          // TODO(dslomov): revisit this. If ijar is not msys-dependent, this is not needed.
          .useDefaultShellEnvironment()
          .addArgument(inputJar.getExecPathString())
          .addArgument(interfaceJar.getExecPathString())
          .setProgressMessage("Extracting interface " + ruleContext.getLabel())
          .setMnemonic("JavaIjar")
          .build(ruleContext));
    }
    return interfaceJar;
  }

  /**
   * Creates a derived artifact from the given artifact by adding the given
   * prefix and removing the extension and replacing it by the given suffix.
   * The new artifact will have the same root as the given one.
   */
  private Artifact derivedArtifact(Artifact artifact, String prefix, String suffix) {
    PathFragment path = artifact.getRootRelativePath();
    String basename = FileSystemUtils.removeExtension(path.getBaseName()) + suffix;
    path = path.replaceName(prefix + basename);
    return ruleContext.getDerivedArtifact(path, artifact.getRoot());
  }
}
