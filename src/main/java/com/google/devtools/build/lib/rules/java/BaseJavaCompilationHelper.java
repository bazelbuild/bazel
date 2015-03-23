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

package com.google.devtools.build.lib.rules.java;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;

/**
 * A helper class for compiling Java targets. This helper does not rely on the
 * presence of rule-specific attributes.
 */
public class BaseJavaCompilationHelper {
  /**
   * Also see DeployArchiveBuilder.SINGLEJAR_MAX_MEMORY. We don't expect that anyone has more
   * than ~500,000 files in a source jar, so 256 MB of memory should be plenty.
   */
  private static final String SINGLEJAR_MAX_MEMORY = "-Xmx256m";

  private final RuleContext ruleContext;

  public BaseJavaCompilationHelper(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Returns the artifacts required to invoke {@code javahome} relative binary
   * in the action.
   */
  public static NestedSet<Artifact> getHostJavabaseInputs(RuleContext ruleContext) {
    // This must have a different name than above, because the middleman creation uses the rule's
    // configuration, although it should use the host configuration.
    return AnalysisUtils.getMiddlemanFor(ruleContext, ":host_jdk");
  }

  private static final ImmutableList<String> SOURCE_JAR_COMMAND_LINE_ARGS = ImmutableList.of(
      "--compression",
      "--normalize",
      "--exclude_build_data",
      "--warn_duplicate_resources");

  private CommandLine sourceJarCommandLine(JavaSemantics semantics, Artifact outputJar,
      Iterable<Artifact> resources, Iterable<Artifact> resourceJars) {
    CustomCommandLine.Builder args = CustomCommandLine.builder();
    args.addExecPath("--output", outputJar);
    args.add(SOURCE_JAR_COMMAND_LINE_ARGS);
    args.addExecPaths("--sources", resourceJars);
    args.add("--resources");
    for (Artifact resource : resources) {
      args.addPaths("%s:%s", resource.getExecPath(),
          semantics.getJavaResourcePath(resource.getRootRelativePath()));
    }
    return args.build();
  }

  /**
   * Creates an Action that packages files into a Jar file.
   *
   * @param semantics delegate semantics for java.
   * @param resources the resources to put into the Jar.
   * @param resourceJars the resource jars to merge into the jar
   * @param outputJar the Jar to create
   */
  public void createSourceJarAction(JavaSemantics semantics, Collection<Artifact> resources,
      Collection<Artifact> resourceJars, Artifact outputJar) {
    ruleContext.registerAction(new SpawnAction.Builder()
        .addOutput(outputJar)
        .addInputs(resources)
        .addInputs(resourceJars)
        .addTransitiveInputs(JavaCompilationHelper.getHostJavabaseInputs(ruleContext))
        .setJarExecutable(
            ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
            ruleContext.getPrerequisiteArtifact("$singlejar", Mode.HOST),
            ImmutableList.of("-client", SINGLEJAR_MAX_MEMORY))
        .setCommandLine(sourceJarCommandLine(semantics, outputJar, resources, resourceJars))
        .useParameterFile(ParameterFileType.SHELL_QUOTED)
        .setProgressMessage("Building source jar " + outputJar.prettyPrint())
        .setMnemonic("JavaSourceJar")
        .build(ruleContext));
  }

  /**
   * Returns the langtools jar Artifact.
   */
  protected final Artifact getLangtoolsJar() {
    return ruleContext.getHostPrerequisiteArtifact("$java_langtools");
  }

  /**
   * Returns the JavaBuilder jar Artifact.
   */
  protected final Artifact getJavaBuilderJar() {
    return ruleContext.getPrerequisiteArtifact("$javabuilder", Mode.HOST);
  }

  /**
   * Returns the instrumentation jar in the given semantics.
   */
  protected final Iterable<Artifact> getInstrumentationJars(JavaSemantics semantics) {
    return semantics.getInstrumentationJars(ruleContext);
  }

  /**
   * Returns the javac bootclasspath artifacts.
   */
  protected final Iterable<Artifact> getBootClasspath() {
    return ruleContext.getPrerequisiteArtifacts("$javac_bootclasspath", Mode.HOST).list();
  }

  private Artifact getIjarArtifact(Artifact jar, boolean addPrefix) {
    if (addPrefix) {
      PathFragment ruleBase = ruleContext.getUniqueDirectory("_ijar");
      PathFragment artifactDirFragment = jar.getRootRelativePath().getParentDirectory();
      String ijarBasename = FileSystemUtils.removeExtension(jar.getFilename()) + "-ijar.jar";
      return getAnalysisEnvironment().getDerivedArtifact(
          ruleBase.getRelative(artifactDirFragment).getRelative(ijarBasename),
          getConfiguration().getGenfilesDirectory());
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
  protected Artifact createIjarAction(final Artifact inputJar, boolean addPrefix) {
    Artifact interfaceJar = getIjarArtifact(inputJar, addPrefix);
    final FilesToRunProvider ijarTarget =
        ruleContext.getExecutablePrerequisite("$ijar", Mode.HOST);
    if (!ruleContext.hasErrors()) {
      ruleContext.registerAction(new SpawnAction.Builder()
          .addInput(inputJar)
          .addOutput(interfaceJar)
          .setExecutable(ijarTarget)
          .addArgument(inputJar.getExecPathString())
          .addArgument(interfaceJar.getExecPathString())
          .setProgressMessage("Extracting interface " + ruleContext.getLabel())
          .setMnemonic("JavaIjar")
          .build(ruleContext));
    }
    return interfaceJar;
  }

  protected final JavaCompileAction.Builder createJavaCompileActionBuilder(
      JavaSemantics semantics) {
    JavaCompileAction.Builder builder = new JavaCompileAction.Builder(ruleContext, semantics);
    builder.setJavaExecutable(
        ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable());
    builder.setJavaBaseInputs(BaseJavaCompilationHelper.getHostJavabaseInputs(ruleContext));
    return builder;
  }

  public RuleContext getRuleContext() {
    return ruleContext;
  }

  public AnalysisEnvironment getAnalysisEnvironment() {
    return ruleContext.getAnalysisEnvironment();
  }

  protected BuildConfiguration getConfiguration() {
    return ruleContext.getConfiguration();
  }

  protected JavaConfiguration getJavaConfiguration() {
    return ruleContext.getFragment(JavaConfiguration.class);
  }

  protected PathFragment outputDir(Artifact outputJar) {
    return workDir(outputJar, "_files");
  }

  /**
   * Produces a derived directory where source files generated by annotation processors should be
   * stored.
   */
  protected PathFragment sourceGenDir(Artifact outputJar) {
    return workDir(outputJar, "_sourcegenfiles");
  }

  protected PathFragment tempDir(Artifact outputJar) {
    return workDir(outputJar, "_temp");
  }

  /**
   * For an output jar and a suffix, produces a derived directory under
   * {@code bin} directory with a given suffix.
   */
  private PathFragment workDir(Artifact outputJar, String suffix) {
    PathFragment path = outputJar.getRootRelativePath();
    String basename = FileSystemUtils.removeExtension(path.getBaseName()) + suffix;
    path = path.replaceName(basename);
    return getConfiguration().getBinDirectory().getExecPath().getRelative(path);
  }

  /**
   * Creates a derived artifact from the given artifact by adding the given
   * prefix and removing the extension and replacing it by the given suffix.
   * The new artifact will have the same root as the given one.
   */
  protected Artifact derivedArtifact(Artifact artifact, String prefix, String suffix) {
    return derivedArtifact(artifact, prefix, suffix, artifact.getRoot());
  }

  protected Artifact derivedArtifact(Artifact artifact, String prefix, String suffix, Root root) {
    PathFragment path = artifact.getRootRelativePath();
    String basename = FileSystemUtils.removeExtension(path.getBaseName()) + suffix;
    path = path.replaceName(prefix + basename);
    return getAnalysisEnvironment().getDerivedArtifact(path, root);
  }
}
