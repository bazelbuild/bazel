// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Action for Java header compilation, to be used if --java_header_compilation is enabled.
 *
 * <p>The header compiler consumes the inputs of a java compilation, and produces an interface jar
 * that can be used as a compile-time jar by upstream targets. The header interface jar is
 * equivalent to the output of ijar, but unlike ijar the header compiler operates directly on Java
 * source files instead post-processing the class outputs of the compilation. Compiling the
 * interface jar from source moves javac off the build's critical path.
 *
 * <p>The implementation of the header compiler tool can be found under {@code
 * //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine}.
 */
public class JavaHeaderCompileAction extends SpawnAction {

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpuIo(/*memoryMb=*/ 750.0, /*cpuUsage=*/ 0.5, /*ioUsage=*/ 0.0);

  /**
   * Constructs an action to compile a set of Java source files to a header interface jar.
   *
   * @param owner the action owner, typically a java_* RuleConfiguredTarget
   * @param tools the set of files comprising the tool that creates the header interface jar
   * @param inputs the set of input artifacts of the compile action
   * @param outputs the outputs of the action
   * @param commandLine the command line arguments for the java header compiler
   * @param progressMessage the message printed during the progression of the build
   */
  protected JavaHeaderCompileAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      CommandLine commandLine,
      String progressMessage) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        LOCAL_RESOURCES,
        commandLine,
        ImmutableMap.<String, String>of(),
        ImmutableSet.<String>of(),
        progressMessage,
        "Turbine");
  }

  /** Builder class to construct Java header compilation actions. */
  public static class Builder {

    private final RuleContext ruleContext;

    private Artifact outputJar;
    @Nullable private Artifact outputDepsProto;
    private final Collection<Artifact> sourceFiles = new ArrayList<>();
    private final Collection<Artifact> sourceJars = new ArrayList<>();
    private NestedSet<Artifact> classpathEntries
        = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private final List<Artifact> bootclasspathEntries = new ArrayList<>();
    @Nullable private String ruleKind;
    @Nullable private Label targetLabel;
    private PathFragment tempDirectory;
    private BuildConfiguration.StrictDepsMode strictJavaDeps
        = BuildConfiguration.StrictDepsMode.OFF;
    private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private final List<Artifact> compileTimeDependencyArtifacts = new ArrayList<>();
    private ImmutableList<String> javacOpts;
    private final List<Artifact> processorPath = new ArrayList<>();
    private final List<String> processorNames = new ArrayList<>();
    private NestedSet<Artifact> javabaseInputs;
    private Artifact javacJar;

    public Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /** Sets the output jdeps file. */
    public Builder setOutputDepsProto(@Nullable Artifact outputDepsProto) {
      this.outputDepsProto = outputDepsProto;
      return this;
    }

    /** Sets the direct dependency artifacts. */
    public Builder setDirectJars(NestedSet<Artifact> directJars) {
      checkNotNull(directJars, "directJars must not be null");
      this.directJars = directJars;
      return this;
    }

    /** Sets the .jdeps artifacts for direct dependencies. */
    public Builder addCompileTimeDependencyArtifacts(
        Collection<Artifact> dependencyArtifacts) {
      checkNotNull(dependencyArtifacts, "dependencyArtifacts must not be null");
      this.compileTimeDependencyArtifacts.addAll(dependencyArtifacts);
      return this;
    }

    /** Sets Java compiler flags. */
    public Builder setJavacOpts(ImmutableList<String> javacOpts) {
      checkNotNull(javacOpts, "javacOpts must not be null");
      this.javacOpts = javacOpts;
      return this;
    }

    /** Sets the output jar. */
    public Builder setOutputJar(Artifact outputJar) {
      checkNotNull(outputJar, "outputJar must not be null");
      this.outputJar = outputJar;
      return this;
    }

    /** Adds a Java source file to compile. */
    public Builder addSourceFile(Artifact sourceFile) {
      checkNotNull(sourceFile, "sourceFile must not be null");
      sourceFiles.add(sourceFile);
      return this;
    }

    /** Adds Java source files to compile. */
    public Builder addSourceFiles(Collection<Artifact> sourceFiles) {
      checkNotNull(sourceFiles, "sourceFiles must not be null");
      this.sourceFiles.addAll(sourceFiles);
      return this;
    }

    /** Adds a jar archive of Java sources to compile. */
    public Builder addSourceJars(Collection<Artifact> sourceJars) {
      checkNotNull(sourceJars, "sourceJars must not be null");
      this.sourceJars.addAll(sourceJars);
      return this;
    }

    /** Sets the compilation classpath entries. */
    public Builder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
      checkNotNull(classpathEntries, "classpathEntries must not be null");
      this.classpathEntries = classpathEntries;
      return this;
    }

    /** Sets the compilation bootclasspath entries. */
    public Builder addAllBootclasspathEntries(
        Collection<Artifact> bootclasspathEntries) {
      checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
      this.bootclasspathEntries.addAll(bootclasspathEntries);
      return this;
    }

    /** Sets the compilation extclasspath entries. */
    public Builder addAllExtClasspathEntries(
        Collection<Artifact> extclasspathEntries) {
      checkNotNull(extclasspathEntries, "extclasspathEntries must not be null");
      // fold extclasspath entries into the bootclasspath; that's what javac ends up doing
      this.bootclasspathEntries.addAll(extclasspathEntries);
      return this;
    }

    /** Sets the annotation processors classpath entries. */
    public Builder addProcessorPaths(Collection<Artifact> processorPaths) {
      checkNotNull(processorPaths, "processorPaths must not be null");
      this.processorPath.addAll(processorPaths);
      return this;
    }

    /** Sets the fully-qualified class names of annotation processors to run. */
    public Builder addProcessorNames(Collection<String> processorNames) {
      checkNotNull(processorNames, "processorNames must not be null");
      this.processorNames.addAll(processorNames);
      return this;
    }

    /** Sets the kind of the build rule being compiled (e.g. {@code java_library}). */
    public Builder setRuleKind(@Nullable String ruleKind) {
      this.ruleKind = ruleKind;
      return this;
    }

    /** Sets the label of the target being compiled. */
    public Builder setTargetLabel(@Nullable Label targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }

    /**
     * Sets the path to a temporary directory, e.g. for extracting sourcejar entries to before
     * compilation.
     */
    public Builder setTempDirectory(PathFragment tempDirectory) {
      checkNotNull(tempDirectory, "tempDirectory must not be null");
      this.tempDirectory = tempDirectory;
      return this;
    }

    /** Sets the Strict Java Deps mode. */
    public Builder setStrictJavaDeps(BuildConfiguration.StrictDepsMode strictJavaDeps) {
      checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
      this.strictJavaDeps = strictJavaDeps;
      return this;
    }

    /** Sets the javabase inputs. */
    public Builder setJavaBaseInputs(NestedSet<Artifact> javabaseInputs) {
      checkNotNull(javabaseInputs, "javabaseInputs must not be null");
      this.javabaseInputs = javabaseInputs;
      return this;
    }

    /** Sets the javac jar. */
    public Builder setJavacJar(Artifact javacJar) {
      checkNotNull(javacJar, "javacJar must not be null");
      this.javacJar = javacJar;
      return this;
    }
    /** Builds and registers the {@link JavaHeaderCompileAction} for a header compilation. */
    public void build(JavaToolchainProvider javaToolchain) {
      checkNotNull(outputDepsProto, "outputDepsProto must not be null");
      checkNotNull(sourceFiles, "sourceFiles must not be null");
      checkNotNull(sourceJars, "sourceJars must not be null");
      checkNotNull(classpathEntries, "classpathEntries must not be null");
      checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
      checkNotNull(tempDirectory, "tempDirectory must not be null");
      checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
      checkNotNull(directJars, "directJars must not be null");
      checkNotNull(compileTimeDependencyArtifacts,
          "compileTimeDependencyArtifacts must not be null");
      checkNotNull(javacOpts, "javacOpts must not be null");
      checkNotNull(processorPath, "processorPath must not be null");
      checkNotNull(processorNames, "processorNames must not be null");

      CommandLine commandLine = buildCommandLine();
      // Invariant: if strictJavaDeps is OFF, then directJars and
      // dependencyArtifacts are ignored
      if (strictJavaDeps == BuildConfiguration.StrictDepsMode.OFF) {
        directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
        compileTimeDependencyArtifacts.clear();
      }
      PathFragment paramFilePath = ParameterFile.derivePath(outputJar.getRootRelativePath());
      Artifact paramsFile = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
              paramFilePath, outputJar.getRoot());
      Action parameterFileWriteAction = new ParameterFileWriteAction(
          ruleContext.getActionOwner(), paramsFile, commandLine,
          ParameterFile.ParameterFileType.UNQUOTED, ISO_8859_1);
      CommandLine turbineCommandLine =
          getBaseArgs(javaToolchain).addPaths("@%s", paramsFile.getExecPath()).build();
      Iterable<Artifact> tools = ImmutableList.of(javacJar, javaToolchain.getHeaderCompiler());
      NestedSet<Artifact> directInputs =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(javabaseInputs)
              .addAll(bootclasspathEntries)
              .addAll(sourceJars)
              .addAll(sourceFiles)
              .addTransitive(directJars)
              .addAll(tools)
              .build();
      NestedSet<Artifact> transitiveInputs =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(directInputs)
              .addTransitive(classpathEntries)
              .addAll(processorPath)
              .addAll(compileTimeDependencyArtifacts)
              .add(paramsFile)
              .build();
      JavaHeaderCompileAction javaHeaderCompileAction =
          new JavaHeaderCompileAction(
              ruleContext.getActionOwner(),
              tools,
              transitiveInputs,
              ImmutableList.of(outputJar, outputDepsProto),
              turbineCommandLine,
              "Compiling Java headers "
                  + outputJar.prettyPrint()
                  + " ("
                  + (sourceFiles.size() + sourceJars.size())
                  + " files)");
      ruleContext.registerAction(parameterFileWriteAction, javaHeaderCompileAction);
    }

    private CustomCommandLine.Builder getBaseArgs(JavaToolchainProvider javaToolchain) {
      return CustomCommandLine.builder()
          .addPath(ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable())
          .add("-Xverify:none")
          .add(javaToolchain.getJvmOptions())
          .addPaths("-Xbootclasspath/p:%s", javacJar.getExecPath())
          .addExecPath("-jar", javaToolchain.getHeaderCompiler());
    }

    /** Builds the header compiler command line. */
    private CommandLine buildCommandLine() {
      CustomCommandLine.Builder result = CustomCommandLine.builder();

      result.addExecPath("--output", outputJar);

      if (outputDepsProto != null) {
        result.addExecPath("--output_deps", outputDepsProto);
      }

      result.add("--temp_dir").addPath(tempDirectory);

      result.addExecPaths("--classpath", classpathEntries);
      result.addExecPaths("--bootclasspath", bootclasspathEntries);

      if (!processorNames.isEmpty()) {
        result.add("--processors", processorNames);
      }
      if (!processorPath.isEmpty()) {
        result.addExecPaths("--processorpath", processorPath);
      }

      result.addExecPaths("--sources", sourceFiles);

      if (!sourceJars.isEmpty()) {
        result.addExecPaths("--source_jars", sourceJars);
      }

      result.add("--javacopts", javacOpts);

      if (ruleKind != null) {
        result.add("--rule_kind");
        result.add(ruleKind);
      }
      if (targetLabel != null) {
        result.add("--target_label");
        if (targetLabel.getPackageIdentifier().getRepository().isDefault()
            || targetLabel.getPackageIdentifier().getRepository().isMain()) {
          result.add(targetLabel.toString());
        } else {
          // @-prefixed strings will be assumed to be params filenames and expanded,
          // so add an extra @ to escape it.
          result.add("@" + targetLabel);
        }
      }

      if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
        result.add(new JavaCompileAction.JarsToTargetsArgv(classpathEntries, directJars));

        if (!compileTimeDependencyArtifacts.isEmpty()) {
          result.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
        }
      }

      return result.build();
    }
  }
}
