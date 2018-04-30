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

import static com.google.common.base.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Action that represents a Java compilation. */
@ThreadCompatible
@Immutable
@AutoCodec
public final class JavaCompileAction extends SpawnAction {
  private static final String JACOCO_INSTRUMENTATION_PROCESSOR = "jacoco";

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpuIo(750 /*MB*/, 0.5 /*CPU*/, 0.0 /*IO*/);

  /** Environment variable that sets the UTF-8 charset. */
  static final ImmutableMap<String, String> UTF8_ENVIRONMENT =
      ImmutableMap.of("LC_CTYPE", "en_US.UTF-8");

  // TODO(#3320): This is missing the configuration's action environment!
  static final ActionEnvironment UTF8_ACTION_ENVIRONMENT =
      ActionEnvironment.create(UTF8_ENVIRONMENT);

  private final CommandLine javaCompileCommandLine;

  /**
   * The directory in which generated classfiles are placed.
   * May be erased/created by the JavaBuilder.
   */
  private final PathFragment classDirectory;

  private final Artifact outputJar;

  /**
   * The list of classpath entries to specify to javac.
   */
  private final NestedSet<Artifact> classpathEntries;

  /** The list of bootclasspath entries to specify to javac. */
  private final ImmutableList<Artifact> bootclasspathEntries;

  /** The list of sourcepath entries to specify to javac. */
  private final ImmutableList<Artifact> sourcePathEntries;

  /**
   * The path to the extdir to specify to javac.
   */
  private final ImmutableList<Artifact> extdirInputs;

  /** The list of classpath entries to search for annotation processors. */
  private final NestedSet<Artifact> processorPath;

  /**
   * The list of annotation processor classes to run.
   */
  private final ImmutableList<String> processorNames;

  /** Set of additional Java source files to compile. */
  private final ImmutableList<Artifact> sourceJars;

  /** The set of explicit Java source files to compile. */
  private final ImmutableSet<Artifact> sourceFiles;

  /**
   * The compiler options to pass to javac.
   */
  private final ImmutableList<String> javacOpts;

  /** The subset of classpath jars provided by direct dependencies. */
  private final NestedSet<Artifact> directJars;

  /**
   * The level of strict dependency checks (off, warnings, or errors).
   */
  private final BuildConfiguration.StrictDepsMode strictJavaDeps;

  /** The tool with which to fix dependency errors. */
  private final String fixDepsTool;

  /** The set of .jdeps artifacts provided by direct dependencies. */
  private final NestedSet<Artifact> compileTimeDependencyArtifacts;

  /**
   * Constructs an action to compile a set of Java source files to class files.
   *
   * @param owner the action owner, typically a java_* RuleConfiguredTarget.
   * @param tools the tools used by the action
   * @param inputs the inputs of the action
   * @param outputs the outputs of the action
   * @param commandLines The invocation command line + possibly the param file command line
   * @param commandLineLimits The command line limits
   * @param javaCompileCommandLine The command line used by the param file
   * @param classDirectory the directory in which generated classfiles are placed
   * @param outputJar the jar file the compilation outputs will be written to
   * @param classpathEntries the compile-time classpath entries
   * @param bootclasspathEntries the compile-time bootclasspath entries
   * @param extdirInputs the compile-time extclasspath entries
   * @param processorPath the classpath to search for annotation processors
   * @param processorNames the annotation processors to run
   * @param sourceJars jars of sources to compile
   * @param sourceFiles source files to compile
   * @param javacOpts the javac options for the compilation
   * @param directJars the subset of classpath jars provided by direct dependencies
   * @param executionInfo the execution info
   * @param strictJavaDeps the Strict Java Deps mode
   * @param fixDepsTool the tool with which to fix dependency errors
   * @param compileTimeDependencyArtifacts the jdeps files for direct dependencies
   * @param progressMessage the progress message
   */
  @VisibleForSerialization
  @AutoCodec.Instantiator
  JavaCompileAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      NestedSet<Artifact> inputs,
      Collection<Artifact> outputs,
      CommandLines commandLines,
      CommandLineLimits commandLineLimits,
      CommandLine javaCompileCommandLine,
      PathFragment classDirectory,
      Artifact outputJar,
      NestedSet<Artifact> classpathEntries,
      ImmutableList<Artifact> bootclasspathEntries,
      ImmutableList<Artifact> sourcePathEntries,
      ImmutableList<Artifact> extdirInputs,
      NestedSet<Artifact> processorPath,
      List<String> processorNames,
      Collection<Artifact> sourceJars,
      ImmutableSet<Artifact> sourceFiles,
      List<String> javacOpts,
      NestedSet<Artifact> directJars,
      Map<String, String> executionInfo,
      StrictDepsMode strictJavaDeps,
      String fixDepsTool,
      NestedSet<Artifact> compileTimeDependencyArtifacts,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        outputJar,
        LOCAL_RESOURCES,
        commandLines,
        commandLineLimits,
        false,
        // TODO(#3320): This is missing the configuration's action environment!
        UTF8_ACTION_ENVIRONMENT,
        ImmutableMap.copyOf(executionInfo),
        progressMessage,
        runfilesSupplier,
        "Javac",
        /* executeUnconditionally= */ false,
        /* extraActionInfoSupplier= */ null);
    this.javaCompileCommandLine = javaCompileCommandLine;
    this.classDirectory = checkNotNull(classDirectory);
    this.outputJar = outputJar;
    this.classpathEntries = classpathEntries;
    this.bootclasspathEntries = ImmutableList.copyOf(bootclasspathEntries);
    this.sourcePathEntries = ImmutableList.copyOf(sourcePathEntries);
    this.extdirInputs = extdirInputs;
    this.processorPath = processorPath;
    this.processorNames = ImmutableList.copyOf(processorNames);
    this.sourceJars = ImmutableList.copyOf(sourceJars);
    this.sourceFiles = sourceFiles;
    this.javacOpts = ImmutableList.copyOf(javacOpts);
    this.directJars = checkNotNull(directJars, "directJars must not be null");
    this.strictJavaDeps = strictJavaDeps;
    this.fixDepsTool = checkNotNull(fixDepsTool);
    this.compileTimeDependencyArtifacts = compileTimeDependencyArtifacts;
  }

  /** Returns the given (passed to constructor) source files. */
  @VisibleForTesting
  ImmutableSet<Artifact> getSourceFiles() {
    return sourceFiles;
  }

  /**
   * Returns the list of paths that represents the classpath.
   */
  @VisibleForTesting
  public Iterable<Artifact> getClasspath() {
    return classpathEntries;
  }

  /** Returns the list of paths that represents the bootclasspath. */
  @VisibleForTesting
  Collection<Artifact> getBootclasspath() {
    return bootclasspathEntries;
  }

  /** Returns the list of paths that represents the sourcepath. */
  @VisibleForTesting
  public Collection<Artifact> getSourcePathEntries() {
    return sourcePathEntries;
  }

  /**
   * Returns the path to the extdir.
   */
  @VisibleForTesting
  public Collection<Artifact> getExtdir() {
    return extdirInputs;
  }

  /**
   * Returns the list of paths that represents the source jars.
   */
  @VisibleForTesting
  public Collection<Artifact> getSourceJars() {
    return sourceJars;
  }

  /** Returns the list of paths that represents the processor path. */
  @VisibleForTesting
  public NestedSet<Artifact> getProcessorpath() {
    return processorPath;
  }

  @VisibleForTesting
  public List<String> getJavacOpts() {
    return javacOpts;
  }

  @VisibleForTesting
  public NestedSet<Artifact> getDirectJars() {
    return directJars;
  }

  @VisibleForTesting
  public NestedSet<Artifact> getCompileTimeDependencyArtifacts() {
    return compileTimeDependencyArtifacts;
  }

  @VisibleForTesting
  public BuildConfiguration.StrictDepsMode getStrictJavaDepsMode() {
    return strictJavaDeps;
  }

  public String getFixDepsTool() {
    return fixDepsTool;
  }

  public PathFragment getClassDirectory() {
    return classDirectory;
  }

  /**
   * Returns the list of class names of processors that should
   * be run.
   */
  @VisibleForTesting
  public List<String> getProcessorNames() {
    return processorNames;
  }

  /**
   * Returns the output jar artifact that gets generated by archiving the results of the Java
   * compilation.
   */
  public Artifact getOutputJar() {
    return outputJar;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return getOutputJar();
  }

  /**
   * Constructs a command line that can be used to invoke the JavaBuilder.
   *
   * <p>Do not use this method, except for testing (and for the in-process strategy).
   */
  @VisibleForTesting
  public Iterable<String> buildCommandLine() {
    try {
      return javaCompileCommandLine.arguments();
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("JavaCompileAction command line expansion cannot fail");
    }
  }

  /** Returns the command and arguments for a java compile action. */
  @VisibleForTesting
  public List<String> getCommand() {
    try {
      return ImmutableList.copyOf(commandLines.getCommandLines().get(0).commandLine.arguments());
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("JavaCompileAction command line expansion cannot fail");
    }
  }

  @Override
  public String toString() {
    try {
      StringBuilder result = new StringBuilder();
      result.append("JavaBuilder ");
      Joiner.on(' ').appendTo(result, javaCompileCommandLine.arguments());
      return result.toString();
    } catch (CommandLineExpansionException e) {
      return "Error expanding command line";
    }
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext) {
    JavaCompileInfo.Builder info = JavaCompileInfo.newBuilder();
    info.addAllSourceFile(Artifact.toExecPaths(getSourceFiles()));
    info.addAllClasspath(Artifact.toExecPaths(getClasspath()));
    info.addAllBootclasspath(Artifact.toExecPaths(getBootclasspath()));
    info.addAllSourcepath(Artifact.toExecPaths(getSourceJars()));
    info.addAllJavacOpt(getJavacOpts());
    info.addAllProcessor(getProcessorNames());
    info.addAllProcessorpath(Artifact.toExecPaths(getProcessorpath()));
    info.setOutputjar(getOutputJar().getExecPathString());

    try {
      return super.getExtraActionInfo(actionKeyContext)
          .setExtension(JavaCompileInfo.javaCompileInfo, info.build());
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("JavaCompileAction command line expansion cannot fail");
    }
  }

  /**
   * Tells {@link Builder} how to create new artifacts. Is there so that {@link Builder} can be
   * exercised in tests without creating a full {@link RuleContext}.
   */
  public interface ArtifactFactory {

    /** Create an artifact with the specified root-relative path under the specified root. */
    Artifact create(PathFragment rootRelativePath, ArtifactRoot root);
  }

  @VisibleForTesting
  static ArtifactFactory createArtifactFactory(final AnalysisEnvironment env) {
    return new ArtifactFactory() {
      @Override
      public Artifact create(PathFragment rootRelativePath, ArtifactRoot root) {
        return env.getDerivedArtifact(rootRelativePath, root);
      }
    };
  }

  /**
   * Builder class to construct Java compile actions.
   */
  public static class Builder {
    private final ActionOwner owner;
    private final AnalysisEnvironment analysisEnvironment;
    private final ArtifactFactory artifactFactory;
    private final BuildConfiguration configuration;
    private final JavaSemantics semantics;

    private PathFragment javaExecutable;
    private List<Artifact> javabaseInputs = ImmutableList.of();
    private Artifact outputJar;
    private Artifact nativeHeaderOutput;
    private Artifact gensrcOutputJar;
    private Artifact manifestProtoOutput;
    private Artifact outputDepsProto;
    private Collection<Artifact> additionalOutputs;
    private Artifact paramFile;
    private Artifact metadata;
    private Artifact artifactForExperimentalCoverage;
    private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
    private final Collection<Artifact> sourceJars = new ArrayList<>();
    private BuildConfiguration.StrictDepsMode strictJavaDeps =
        BuildConfiguration.StrictDepsMode.OFF;
    private String fixDepsTool = "add_dep";
    private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private NestedSet<Artifact> compileTimeDependencyArtifacts =
        NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    private List<String> javacOpts = new ArrayList<>();
    private ImmutableList<String> javacJvmOpts = ImmutableList.of();
    private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
    private boolean compressJar;
    private NestedSet<Artifact> classpathEntries =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private ImmutableList<Artifact> bootclasspathEntries = ImmutableList.of();
    private ImmutableList<Artifact> sourcePathEntries = ImmutableList.of();
    private ImmutableList<Artifact> extdirInputs = ImmutableList.of();
    private FilesToRunProvider javaBuilder;
    private Artifact langtoolsJar;
    private NestedSet<Artifact> toolsJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private ImmutableList<Artifact> instrumentationJars = ImmutableList.of();
    private PathFragment sourceGenDirectory;
    private PathFragment tempDirectory;
    private PathFragment classDirectory;
    private NestedSet<Artifact> processorPath = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private final List<String> processorNames = new ArrayList<>();
    private Label targetLabel;
    @Nullable private String injectingRuleKind;

    /**
     * Creates a Builder from an owner and a build configuration.
     */
    public Builder(ActionOwner owner, AnalysisEnvironment analysisEnvironment,
        ArtifactFactory artifactFactory, BuildConfiguration configuration,
        JavaSemantics semantics) {
      this.owner = owner;
      this.analysisEnvironment = analysisEnvironment;
      this.artifactFactory = artifactFactory;
      this.configuration = configuration;
      this.semantics = semantics;
    }

    /**
     * Creates a Builder from an owner and a build configuration.
     */
    public Builder(final RuleContext ruleContext, JavaSemantics semantics) {
      this(
          ruleContext.getActionOwner(),
          ruleContext.getAnalysisEnvironment(),
          new ArtifactFactory() {
            @Override
            public Artifact create(PathFragment rootRelativePath, ArtifactRoot root) {
              return ruleContext.getDerivedArtifact(rootRelativePath, root);
            }
          },
          ruleContext.getConfiguration(),
          semantics);
    }

    public JavaCompileAction build() {
      // TODO(bazel-team): all the params should be calculated before getting here, and the various
      // aggregation code below should go away.
      final String pathSeparator = configuration.getHostPathSeparator();
      final List<String> internedJcopts = new ArrayList<>();
      for (String jcopt : javacOpts) {
        internedJcopts.add(StringCanonicalizer.intern(jcopt));
      }

      // Invariant: if strictJavaDeps is OFF, then directJars and
      // dependencyArtifacts are ignored
      if (strictJavaDeps == BuildConfiguration.StrictDepsMode.OFF) {
        directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
        compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      }

      // Invariant: if java_classpath is set to 'off', dependencyArtifacts are ignored
      JavaConfiguration javaConfiguration = configuration.getFragment(JavaConfiguration.class);
      if (javaConfiguration.getReduceJavaClasspath() == JavaClasspathMode.OFF) {
        compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      }

      Preconditions.checkState(javaExecutable != null, owner);

      ImmutableList.Builder<Artifact> outputsBuilder = ImmutableList.<Artifact>builder();
      Stream.of(
              outputJar,
              metadata,
              gensrcOutputJar,
              manifestProtoOutput,
              outputDepsProto,
              nativeHeaderOutput)
          .filter(x -> x != null)
          .forEachOrdered(outputsBuilder::add);
      if (additionalOutputs != null) {
        outputsBuilder.addAll(additionalOutputs);
      }
      ImmutableList<Artifact> outputs = outputsBuilder.build();

      // The actual params-file-based command line executed for a compile action.
      CustomCommandLine.Builder javaBuilderCommandLine = CustomCommandLine.builder();
      Artifact javaBuilderJar = checkNotNull(javaBuilder.getExecutable());
      if (!javaBuilderJar.getExtension().equals("jar")) {
        // JavaBuilder is a non-deploy.jar executable.
        javaBuilderCommandLine.addExecPath(javaBuilderJar);
      } else {
        javaBuilderCommandLine.addPath(javaExecutable).addAll(javacJvmOpts);
        if (!instrumentationJars.isEmpty()) {
          javaBuilderCommandLine
              .addExecPaths(
                  "-cp",
                  VectorArg.join(pathSeparator)
                      .each(
                          ImmutableList.<Artifact>builder()
                              .addAll(instrumentationJars)
                              .add(javaBuilderJar)
                              .build()))
              .addDynamicString(semantics.getJavaBuilderMainClass());
        } else {
          // If there are no instrumentation jars, use simpler '-jar' option to launch JavaBuilder.
          javaBuilderCommandLine.addExecPath("-jar", javaBuilderJar);
        }
      }

      if (artifactForExperimentalCoverage != null) {
        analysisEnvironment.registerAction(new LazyWritePathsFileAction(
            owner, artifactForExperimentalCoverage, sourceFiles, false));
      }

      NestedSet<Artifact> tools =
          NestedSetBuilder.<Artifact>stableOrder()
              .add(langtoolsJar)
              .addTransitive(toolsJars)
              .addTransitive(javaBuilder.getFilesToRun())
              .addAll(instrumentationJars)
              .build();

      NestedSetBuilder<Artifact> inputsBuilder =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(classpathEntries)
              .addTransitive(compileTimeDependencyArtifacts)
              .addTransitive(processorPath)
              .addAll(sourceJars)
              .addAll(sourceFiles)
              .addAll(javabaseInputs)
              .addAll(bootclasspathEntries)
              .addAll(sourcePathEntries)
              .addAll(extdirInputs)
              .addTransitive(tools);
      if (artifactForExperimentalCoverage != null) {
        inputsBuilder.add(artifactForExperimentalCoverage);
      }

      final CommandLines commandLines;
      CustomCommandLine javaCompileCommandLine = buildParamFileContents(internedJcopts);
      if (configuration.deferParamFiles()) {
        commandLines =
            CommandLines.builder()
                .addCommandLine(javaBuilderCommandLine.build())
                .addCommandLine(
                    javaCompileCommandLine,
                    ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
                        .setCharset(ISO_8859_1)
                        .setUseAlways(true)
                        .build())
                .build();
      } else {
        if (paramFile == null) {
          paramFile =
              artifactFactory.create(
                  ParameterFile.derivePath(outputJar.getRootRelativePath()),
                  configuration.getBinDirectory(
                      targetLabel.getPackageIdentifier().getRepository()));
        }
        inputsBuilder.add(paramFile);
        javaBuilderCommandLine.addFormatted("@%s", paramFile.getExecPath());
        commandLines = CommandLines.of(javaBuilderCommandLine.build());
        Action parameterFileWriteAction =
            new ParameterFileWriteAction(
                owner,
                paramFile,
                javaCompileCommandLine,
                ParameterFile.ParameterFileType.UNQUOTED,
                ISO_8859_1);
        analysisEnvironment.registerAction(parameterFileWriteAction);
      }

      NestedSet<Artifact> inputs = inputsBuilder.build();
      return new JavaCompileAction(
          owner,
          tools,
          inputs,
          outputs,
          commandLines,
          configuration.getCommandLineLimits(),
          javaCompileCommandLine,
          classDirectory,
          outputJar,
          classpathEntries,
          bootclasspathEntries,
          sourcePathEntries,
          extdirInputs,
          processorPath,
          processorNames,
          sourceJars,
          sourceFiles,
          internedJcopts,
          directJars,
          executionInfo,
          strictJavaDeps,
          fixDepsTool,
          compileTimeDependencyArtifacts,
          getProgressMessage(),
          javaBuilder.getRunfilesSupplier());
    }

    private CustomCommandLine buildParamFileContents(Collection<String> javacOpts) {
      checkNotNull(classDirectory, "classDirectory should not be null");
      checkNotNull(tempDirectory, "tempDirectory should not be null");

      CustomCommandLine.Builder result = CustomCommandLine.builder();

      result.add("--classdir").addPath(classDirectory);
      result.add("--tempdir").addPath(tempDirectory);
      if (outputJar != null) {
        result.addExecPath("--output", outputJar);
      }
      if (nativeHeaderOutput != null) {
        result.addExecPath("--native_header_output", nativeHeaderOutput);
      }
      if (sourceGenDirectory != null) {
        result.add("--sourcegendir").addPath(sourceGenDirectory);
      }
      if (gensrcOutputJar != null) {
        result.addExecPath("--generated_sources_output", gensrcOutputJar);
      }
      if (manifestProtoOutput != null) {
        result.addExecPath("--output_manifest_proto", manifestProtoOutput);
      }
      if (compressJar) {
        result.add("--compress_jar");
      }
      if (outputDepsProto != null) {
        result.addExecPath("--output_deps_proto", outputDepsProto);
      }
      if (!extdirInputs.isEmpty()) {
        result.addExecPaths("--extclasspath", extdirInputs);
      }
      if (!bootclasspathEntries.isEmpty()) {
        result.addExecPaths("--bootclasspath", bootclasspathEntries);
      }
      if (!sourcePathEntries.isEmpty()) {
        result.addExecPaths("--sourcepath", sourcePathEntries);
      }
      if (!processorPath.isEmpty()) {
        result.addExecPaths("--processorpath", processorPath);
      }
      if (!processorNames.isEmpty()) {
        result.addAll("--processors", ImmutableList.copyOf(processorNames));
      }
      if (!sourceJars.isEmpty()) {
        result.addExecPaths("--source_jars", ImmutableList.copyOf(sourceJars));
      }
      if (!sourceFiles.isEmpty()) {
        result.addExecPaths("--sources", sourceFiles);
      }
      if (!javacOpts.isEmpty()) {
        result.addAll("--javacopts", ImmutableList.copyOf(javacOpts));
        // terminate --javacopts with `--` to support javac flags that start with `--`
        result.add("--");
      }
      if (targetLabel != null) {
        result.add("--target_label");
        if (targetLabel.getPackageIdentifier().getRepository().isDefault()
            || targetLabel.getPackageIdentifier().getRepository().isMain()) {
          result.addLabel(targetLabel);
        } else {
          // @-prefixed strings will be assumed to be filenames and expanded by
          // {@link JavaLibraryBuildRequest}, so add an extra &at; to escape it.
          result.addPrefixedLabel("@", targetLabel);
        }
      }
      if (injectingRuleKind != null) {
        result.add("--injecting_rule_kind", injectingRuleKind);
      }

      if (!classpathEntries.isEmpty()) {
        result.addExecPaths("--classpath", classpathEntries);
      }

      // strict_java_deps controls whether the mapping from jars to targets is
      // written out and whether we try to minimize the compile-time classpath.
      if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
        result.add("--strict_java_deps", strictJavaDeps.toString());
        result.addExecPaths("--direct_dependencies", directJars);

        if (configuration.getFragment(JavaConfiguration.class).getReduceJavaClasspath()
            == JavaClasspathMode.JAVABUILDER) {
          result.add("--reduce_classpath");

          if (!compileTimeDependencyArtifacts.isEmpty()) {
            result.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
          }
        }
      }
      result.add("--experimental_fix_deps_tool", fixDepsTool);

      // Chose what artifact to pass to JavaBuilder, as input to jacoco instrumentation processor.
      // metadata should be null when --experimental_java_coverage is true.
      Artifact coverageArtifact = metadata != null ? metadata : artifactForExperimentalCoverage;
      if (coverageArtifact != null) {
        result.add("--post_processor");
        result.addExecPath(JACOCO_INSTRUMENTATION_PROCESSOR, coverageArtifact);
        result.addPath(
            configuration
                .getCoverageMetadataDirectory(targetLabel.getPackageIdentifier().getRepository())
                .getExecPath());
        result.add("-*Test");
        result.add("-*TestCase");
      }
      return result.build();
    }

    private LazyString getProgressMessage() {
      Artifact outputJar = this.outputJar;
      int sourceFileCount = sourceFiles.size();
      int sourceJarCount = sourceJars.size();
      String annotationProcessorNames = getProcessorNames();
      return new LazyString() {
        @Override
        public String toString() {
          StringBuilder sb = new StringBuilder("Building ");
          sb.append(outputJar.prettyPrint());
          sb.append(" (");
          boolean first = true;
          first = appendCount(sb, first, sourceFileCount, "source file");
          first = appendCount(sb, first, sourceJarCount, "source jar");
          sb.append(")");
          sb.append(annotationProcessorNames);
          return sb.toString();
        }
      };
    }

    private String getProcessorNames() {
      if (processorNames.isEmpty()) {
        return "";
      }
      StringBuilder sb = new StringBuilder();
      List<String> shortNames = new ArrayList<>();
      for (String name : processorNames) {
        // Annotation processor names are qualified class names. Omit the package part for the
        // progress message, e.g. `com.google.Foo` -> `Foo`.
        int idx = name.lastIndexOf('.');
        String shortName = idx != -1 ? name.substring(idx + 1) : name;
        shortNames.add(shortName);
      }
      sb.append(" and running annotation processors (");
      Joiner.on(", ").appendTo(sb, shortNames);
      sb.append(")");
      return sb.toString();
    }

    /**
     * Append an input count to the progress message, e.g. "2 source jars". If an input count has
     * already been appended, prefix with ", ".
     */
    private static boolean appendCount(StringBuilder sb, boolean first, int count, String name) {
      if (count > 0) {
        if (!first) {
          sb.append(", ");
        } else {
          first = false;
        }
        sb.append(count).append(' ').append(name);
        if (count > 1) {
          sb.append('s');
        }
      }
      return first;
    }

    public Builder setJavaExecutable(PathFragment javaExecutable) {
      this.javaExecutable = javaExecutable;
      return this;
    }

    public Builder setJavaBaseInputs(Iterable<Artifact> javabaseInputs) {
      this.javabaseInputs = ImmutableList.copyOf(javabaseInputs);
      return this;
    }

    public Builder setOutputJar(Artifact outputJar) {
      this.outputJar = outputJar;
      return this;
    }

    public Builder setNativeHeaderOutput(Artifact nativeHeaderOutput) {
      this.nativeHeaderOutput = nativeHeaderOutput;
      return this;
    }

    public Builder setGensrcOutputJar(Artifact gensrcOutputJar) {
      this.gensrcOutputJar = gensrcOutputJar;
      return this;
    }

    public Builder setManifestProtoOutput(Artifact manifestProtoOutput) {
      this.manifestProtoOutput = manifestProtoOutput;
      return this;
    }

    public Builder setOutputDepsProto(Artifact outputDepsProto) {
      this.outputDepsProto = outputDepsProto;
      return this;
    }

    public Builder setAdditionalOutputs(Collection<Artifact> outputs) {
      this.additionalOutputs = outputs;
      return this;
    }

    public Builder setMetadata(Artifact metadata) {
      this.metadata = metadata;
      return this;
    }

    public Builder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
      this.sourceFiles = sourceFiles;
      return this;
    }

    public Builder addSourceJars(Collection<Artifact> sourceJars) {
      this.sourceJars.addAll(sourceJars);
      return this;
    }

    /**
     * Sets the strictness of Java dependency checking, see {@link
     * com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode}.
     */
    public Builder setStrictJavaDeps(BuildConfiguration.StrictDepsMode strictDeps) {
      strictJavaDeps = strictDeps;
      return this;
    }

    /** Sets the tool with which to fix dependency errors. */
    public Builder setFixDepsTool(String depsTool) {
      fixDepsTool = depsTool;
      return this;
    }

    /** Accumulates the given jar artifacts as being provided by direct dependencies. */
    public Builder setDirectJars(NestedSet<Artifact> directJars) {
      this.directJars = checkNotNull(directJars, "directJars must not be null");
      return this;
    }

    public Builder setCompileTimeDependencyArtifacts(NestedSet<Artifact> dependencyArtifacts) {
      checkNotNull(compileTimeDependencyArtifacts, "dependencyArtifacts must not be null");
      this.compileTimeDependencyArtifacts = dependencyArtifacts;
      return this;
    }

    public Builder setJavacOpts(Iterable<String> copts) {
      this.javacOpts = ImmutableList.copyOf(copts);
      return this;
    }

    public Builder setJavacJvmOpts(ImmutableList<String> opts) {
      this.javacJvmOpts = opts;
      return this;
    }

    public Builder setJavacExecutionInfo(ImmutableMap<String, String> executionInfo) {
      this.executionInfo = executionInfo;
      return this;
    }

    public Builder setCompressJar(boolean compressJar) {
      this.compressJar = compressJar;
      return this;
    }

    public Builder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
      this.classpathEntries = classpathEntries;
      return this;
    }

    public Builder setBootclasspathEntries(Iterable<Artifact> bootclasspathEntries) {
      this.bootclasspathEntries = ImmutableList.copyOf(bootclasspathEntries);
      return this;
    }

    public Builder setSourcePathEntries(Iterable<Artifact> sourcePathEntries) {
      this.sourcePathEntries = ImmutableList.copyOf(sourcePathEntries);
      return this;
    }

    public Builder setExtdirInputs(Iterable<Artifact> extdirEntries) {
      this.extdirInputs = ImmutableList.copyOf(extdirEntries);
      return this;
    }

    /**
     * Sets the directory where source files generated by annotation processors should be stored.
     */
    public Builder setSourceGenDirectory(PathFragment sourceGenDirectory) {
      this.sourceGenDirectory = sourceGenDirectory;
      return this;
    }

    public Builder setTempDirectory(PathFragment tempDirectory) {
      this.tempDirectory = tempDirectory;
      return this;
    }

    public Builder setClassDirectory(PathFragment classDirectory) {
      this.classDirectory = classDirectory;
      return this;
    }

    public Builder setProcessorPaths(NestedSet<Artifact> processorPaths) {
      this.processorPath = processorPaths;
      return this;
    }

    public Builder addProcessorNames(Collection<String> processorNames) {
      this.processorNames.addAll(processorNames);
      return this;
    }

    public Builder setLangtoolsJar(Artifact langtoolsJar) {
      this.langtoolsJar = langtoolsJar;
      return this;
    }

    /** Sets the tools jars. */
    public Builder setToolsJars(NestedSet<Artifact> toolsJars) {
      checkNotNull(toolsJars, "toolsJars must not be null");
      this.toolsJars = toolsJars;
      return this;
    }

    public Builder setJavaBuilder(FilesToRunProvider javaBuilder) {
      this.javaBuilder = javaBuilder;
      return this;
    }

    public Builder setInstrumentationJars(Iterable<Artifact> instrumentationJars) {
      this.instrumentationJars = ImmutableList.copyOf(instrumentationJars);
      return this;
    }

    public Builder setArtifactForExperimentalCoverage(Artifact artifactForExperimentalCoverage) {
      this.artifactForExperimentalCoverage = artifactForExperimentalCoverage;
      return this;
    }

    public Builder setTargetLabel(Label targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }

    public Builder setInjectingRuleKind(@Nullable String injectingRuleKind) {
      this.injectingRuleKind = injectingRuleKind;
      return this;
    }
  }
}
