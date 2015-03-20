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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomArgv;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomMultiArgv;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.ImmutableIterable;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Action that represents a Java compilation.
 */
@ThreadCompatible
public class JavaCompileAction extends AbstractAction {

  private static final String GUID = "786e174d-ed97-4e79-9f61-ae74430714cf";

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpuIo(750 /*MB*/, 0.5 /*CPU*/, 0.0 /*IO*/);

  private final CommandLine javaCompileCommandLine;
  private final CommandLine commandLine;

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

  /**
   * The list of classpath entries to search for annotation processors.
   */
  private final ImmutableList<Artifact> processorPath;

  /**
   * The list of annotation processor classes to run.
   */
  private final ImmutableList<String> processorNames;

  /**
   * The translation messages.
   */
  private final ImmutableList<Artifact> messages;

  /**
   * The set of resources to put into the jar.
   */
  private final ImmutableList<Artifact> resources;

  /**
   * The set of classpath resources to put into the jar.
   */
  private final ImmutableList<Artifact> classpathResources;

  /**
   * The set of files which contain lists of additional Java source files to
   * compile.
   */
  private final ImmutableList<Artifact> sourceJars;

  /**
   * The set of explicit Java source files to compile.
   */
  private final ImmutableList<Artifact> sourceFiles;

  /**
   * The compiler options to pass to javac.
   */
  private final ImmutableList<String> javacOpts;

  /**
   * The subset of classpath jars provided by direct dependencies.
   */
  private final ImmutableList<Artifact> directJars;

  /**
   * The level of strict dependency checks (off, warnings, or errors).
   */
  private final BuildConfiguration.StrictDepsMode strictJavaDeps;

  /**
   * The set of .deps artifacts provided by direct dependencies.
   */
  private final ImmutableList<Artifact> compileTimeDependencyArtifacts;

  /**
   * The java semantics to get the list of action outputs.
   */
  private final JavaSemantics semantics;

  /**
   * Constructs an action to compile a set of Java source files to class files.
   *
   * @param owner the action owner, typically a java_* RuleConfiguredTarget.
   * @param baseInputs the set of the input artifacts of the compile action
   *        without the parameter file action;
   * @param outputs the outputs of the action
   * @param javaCompileCommandLine the command line for the java library
   *        builder - it's actually written to the parameter file, but other
   *        parts (for example, ide_build_info) need access to the data
   * @param commandLine the actual invocation command line
   */
  private JavaCompileAction(ActionOwner owner,
                            Iterable<Artifact> baseInputs,
                            Collection<Artifact> outputs,
                            CommandLine javaCompileCommandLine,
                            CommandLine commandLine,
                            PathFragment classDirectory,
                            Artifact outputJar,
                            NestedSet<Artifact> classpathEntries,
                            List<Artifact> processorPath,
                            Artifact langtoolsJar,
                            Artifact javaBuilderJar,
                            Iterable<Artifact> instrumentationJars,
                            List<String> processorNames,
                            Collection<Artifact> messages,
                            Collection<Artifact> resources,
                            Collection<Artifact> classpathResources,
                            Collection<Artifact> sourceJars,
                            Collection<Artifact> sourceFiles,
                            List<String> javacOpts,
                            Collection<Artifact> directJars,
                            BuildConfiguration.StrictDepsMode strictJavaDeps,
                            Collection<Artifact> compileTimeDependencyArtifacts,
                            JavaSemantics semantics) {
    super(owner, NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(classpathEntries)
            .addAll(processorPath)
            .addAll(messages)
            .addAll(resources)
            .addAll(classpathResources)
            .addAll(sourceJars)
            .addAll(sourceFiles)
            .addAll(compileTimeDependencyArtifacts)
            .addAll(baseInputs)
            .add(langtoolsJar)
            .add(javaBuilderJar)
            .addAll(instrumentationJars)
            .build(),
        outputs);
    this.javaCompileCommandLine = javaCompileCommandLine;
    this.commandLine = commandLine;

    this.classDirectory = Preconditions.checkNotNull(classDirectory);
    this.outputJar = outputJar;
    this.classpathEntries = classpathEntries;
    this.processorPath = ImmutableList.copyOf(processorPath);
    this.processorNames = ImmutableList.copyOf(processorNames);
    this.messages = ImmutableList.copyOf(messages);
    this.resources = ImmutableList.copyOf(resources);
    this.classpathResources = ImmutableList.copyOf(classpathResources);
    this.sourceJars = ImmutableList.copyOf(sourceJars);
    this.sourceFiles = ImmutableList.copyOf(sourceFiles);
    this.javacOpts = ImmutableList.copyOf(javacOpts);
    this.directJars = ImmutableList.copyOf(directJars);
    this.strictJavaDeps = strictJavaDeps;
    this.compileTimeDependencyArtifacts = ImmutableList.copyOf(compileTimeDependencyArtifacts);
    this.semantics = semantics;
  }

  /**
   * Returns the given (passed to constructor) source files.
   */
  @VisibleForTesting
  public Collection<Artifact> getSourceFiles() {
    return sourceFiles;
  }

  /**
   * Returns the list of paths that represent the resources to be added to the
   * jar.
   */
  @VisibleForTesting
  public Collection<Artifact> getResources() {
    return resources;
  }

  /**
   * Returns the list of paths that represents the classpath.
   */
  @VisibleForTesting
  public Iterable<Artifact> getClasspath() {
    return classpathEntries;
  }

  /**
   * Returns the list of paths that represents the source jars.
   */
  @VisibleForTesting
  public Collection<Artifact> getSourceJars() {
    return sourceJars;
  }

  /**
   * Returns the list of paths that represents the processor path.
   */
  @VisibleForTesting
  public List<Artifact> getProcessorpath() {
    return processorPath;
  }

  @VisibleForTesting
  public List<String> getJavacOpts() {
    return javacOpts;
  }

  @VisibleForTesting
  public Collection<Artifact> getDirectJars() {
    return directJars;
  }

  @VisibleForTesting
  public Collection<Artifact> getCompileTimeDependencyArtifacts() {
    return compileTimeDependencyArtifacts;
  }

  @VisibleForTesting
  public BuildConfiguration.StrictDepsMode getStrictJavaDepsMode() {
    return strictJavaDeps;
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
   * Returns the output jar artifact that gets generated by archiving the
   * results of the Java compilation and the declared resources.
   */
  public Artifact getOutputJar() {
    return outputJar;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return getOutputJar();
  }

  /**
   * Constructs a command line that can be used to invoke the
   * JavaBuilder.
   *
   * <p>Do not use this method, except for testing (and for the in-process
   * strategy).
   */
  @VisibleForTesting
  public Iterable<String> buildCommandLine() {
    return javaCompileCommandLine.arguments();
  }

  /**
   * Returns the command and arguments for a java compile action.
   */
  public List<String> getCommand() {
    return ImmutableList.copyOf(commandLine.arguments());
  }

  @Override
  @ThreadCompatible
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    try {
      List<ActionInput> outputs = new ArrayList<>();
      outputs.addAll(getOutputs());
      // Add a few useful side-effect output files to the list to retrieve.
      // TODO(bazel-team): Just make these Artifacts.
      PathFragment classDirectory = getClassDirectory();
      outputs.addAll(semantics.getExtraJavaCompileOutputs(classDirectory));
      outputs.add(ActionInputHelper.fromPath(classDirectory.getChild("srclist").getPathString()));

      try {
        // Make sure the directories exist, else the distributor will bomb.
        Path classDirectoryPath = executor.getExecRoot().getRelative(getClassDirectory());
        FileSystemUtils.createDirectoryAndParents(classDirectoryPath);
      } catch (IOException e) {
        throw new EnvironmentalExecException(e.getMessage());
      }

      final ImmutableList<ActionInput> finalOutputs = ImmutableList.copyOf(outputs);
      Spawn spawn = new BaseSpawn(getCommand(), ImmutableMap.<String, String>of(),
          ImmutableMap.<String, String>of(), this, LOCAL_RESOURCES) {
        @Override
        public Collection<? extends ActionInput> getOutputFiles() {
          return finalOutputs;
        }
      };

      executor.getSpawnActionContext(getMnemonic()).exec(spawn, actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException("Java compilation in rule '" + getOwner().getLabel() + "'",
          executor.getVerboseFailures(), this);
    }
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addStrings(commandLine.arguments());
    return f.hexDigestAndReset();
  }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    for (String arg : ShellEscaper.escapeAll(commandLine.arguments())) {
      message.append("  Command-line argument: ");
      message.append(arg);
      message.append('\n');
    }
    return message.toString();
  }

  @Override
  public String getMnemonic() {
    return "Javac";
  }

  @Override
  protected String getRawProgressMessage() {
    int count = sourceFiles.size();
    if (count == 0) { // nothing to compile, just bundling resources and messages
      count = resources.size() + classpathResources.size() + messages.size();
    }
    return "Building " + outputJar.prettyPrint() + " (" + count + " files)";
  }

  @Override
  public String describeStrategy(Executor executor) {
    return getContext(executor).strategyLocality(getMnemonic(), true);
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    SpawnActionContext context = getContext(executor);
    if (context.isRemotable(getMnemonic(), true)) {
      return ResourceSet.ZERO;
    }
    return LOCAL_RESOURCES;
  }

  protected SpawnActionContext getContext(Executor executor) {
    return executor.getSpawnActionContext(getMnemonic());
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append("JavaBuilder ");
    Joiner.on(' ').appendTo(result, commandLine.arguments());
    return result.toString();
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    JavaCompileInfo.Builder info = JavaCompileInfo.newBuilder();
    info.addAllSourceFile(Artifact.toExecPaths(getSourceFiles()));
    info.addAllClasspath(Artifact.toExecPaths(getClasspath()));
    info.addClasspath(getClassDirectory().getPathString());
    info.addAllSourcepath(Artifact.toExecPaths(getSourceJars()));
    info.addAllJavacOpt(getJavacOpts());
    info.addAllProcessor(getProcessorNames());
    info.addAllProcessorpath(Artifact.toExecPaths(getProcessorpath()));
    info.setOutputjar(getOutputJar().getExecPathString());

    return super.getExtraActionInfo()
        .setExtension(JavaCompileInfo.javaCompileInfo, info.build());
  }

  /**
   * Creates an instance.
   *
   * @param configuration the build configuration, which provides the default options and the path
   *        to the compiler, etc.
   * @param classDirectory the directory in which generated classfiles are placed relative to the
   *        exec root
   * @param sourceGenDirectory the directory where source files generated by annotation processors
   *        should be stored.
   * @param tempDirectory a directory in which the library builder can store temporary files
   *        relative to the exec root
   * @param outputJar output jar
   * @param compressJar if true compress the output jar
   * @param outputDepsProto the proto file capturing dependency information
   * @param classpath the complete classpath, the directory in which generated classfiles are placed
   * @param processorPath the classpath where javac should search for annotation processors
   * @param processorNames the classes that javac should use as annotation processors
   * @param messages the message files for translation
   * @param resources the set of resources to put into the jar
   * @param classpathResources the set of classpath resources to put into the jar
   * @param sourceJars the set of jars containing additional source files to compile
   * @param sourceFiles the set of explicit Java source files to compile
   * @param javacOpts the compiler options to pass to javac
   */
  private static CustomCommandLine.Builder javaCompileCommandLine(
      final JavaSemantics semantics,
      final BuildConfiguration configuration,
      final PathFragment classDirectory,
      final PathFragment sourceGenDirectory,
      PathFragment tempDirectory,
      Artifact outputJar,
      Artifact gensrcOutputJar,
      boolean compressJar,
      Artifact outputDepsProto,
      final NestedSet<Artifact> classpath,
      List<Artifact> processorPath,
      List<String> processorNames,
      Collection<Artifact> messages,
      Collection<Artifact> resources,
      Collection<Artifact> classpathResources,
      Collection<Artifact> sourceJars,
      Collection<Artifact> sourceFiles,
      List<String> javacOpts,
      final Collection<Artifact> directJars,
      BuildConfiguration.StrictDepsMode strictJavaDeps,
      Collection<Artifact> compileTimeDependencyArtifacts,
      String ruleKind,
      Label targetLabel) {
    Preconditions.checkNotNull(classDirectory);
    Preconditions.checkNotNull(tempDirectory);

    CustomCommandLine.Builder result = CustomCommandLine.builder();

    result.add("--classdir").addPath(classDirectory);

    result.add("--tempdir").addPath(tempDirectory);

    if (outputJar != null) {
      result.addExecPath("--output", outputJar);
    }

    if (gensrcOutputJar != null) {
      result.add("--sourcegendir").addPath(sourceGenDirectory);
      result.addExecPath("--generated_sources_output", gensrcOutputJar);
    }

    if (compressJar) {
      result.add("--compress_jar");
    }

    if (outputDepsProto != null) {
      result.addExecPath("--output_deps_proto", outputDepsProto);
    }

    result.add("--classpath").add(new CustomArgv() {
      @Override
      public String argv() {
        List<PathFragment> classpathEntries = new ArrayList<>();
        for (Artifact classpathArtifact : classpath) {
          classpathEntries.add(classpathArtifact.getExecPath());
        }
        classpathEntries.add(classDirectory);
        return Joiner.on(configuration.getHostPathSeparator()).join(classpathEntries);
      }
    });

    if (!processorPath.isEmpty()) {
      result.addJoinExecPaths("--processorpath",
          configuration.getHostPathSeparator(), processorPath);
    }

    if (!processorNames.isEmpty()) {
      result.add("--processors", processorNames);
    }

    if (!messages.isEmpty()) {
      result.add("--messages");
      for (Artifact message : messages) {
        addAsResourcePrefixedExecPath(semantics, message, result);
      }
    }

    if (!resources.isEmpty()) {
      result.add("--resources");
      for (Artifact resource : resources) {
        addAsResourcePrefixedExecPath(semantics, resource, result);
      }
    }

    if (!classpathResources.isEmpty()) {
      result.addExecPaths("--classpath_resources", classpathResources);
    }

    if (!sourceJars.isEmpty()) {
      result.addExecPaths("--source_jars", sourceJars);
    }

    result.addExecPaths("--sources", sourceFiles);

    if (!javacOpts.isEmpty()) {
      result.add("--javacopts", javacOpts);
    }

    // strict_java_deps controls whether the mapping from jars to targets is
    // written out and whether we try to minimize the compile-time classpath.
    if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
      result.add("--strict_java_deps");
      result.add((semantics.useStrictJavaDeps(configuration) ? strictJavaDeps
          : BuildConfiguration.StrictDepsMode.OFF).toString());
      result.add(new CustomMultiArgv() {
        @Override
        public Iterable<String> argv() {
          return addJarsToTargets(classpath, directJars);
        }
      });

      if (configuration.getFragment(JavaConfiguration.class).getReduceJavaClasspath()
          == JavaClasspathMode.JAVABUILDER) {
        result.add("--reduce_classpath");

        if (!compileTimeDependencyArtifacts.isEmpty()) {
          result.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
        }
      }
    }

    if (ruleKind != null) {
      result.add("--rule_kind");
      result.add(ruleKind);
    }
    if (targetLabel != null) {
      result.add("--target_label");
      if (targetLabel.getPackageIdentifier().getRepository().isDefault()) {
        result.add(targetLabel.toString());
      } else {
        // @-prefixed strings will be assumed to be filenames and expanded by
        // {@link JavaLibraryBuildRequest}, so add an extra &at; to escape it.
        result.add("@" + targetLabel);
      }
    }

    return result;
  }

  private static void addAsResourcePrefixedExecPath(JavaSemantics semantics,
      Artifact artifact, CustomCommandLine.Builder builder) {
    PathFragment execPath = artifact.getExecPath();
    PathFragment resourcePath = semantics.getJavaResourcePath(artifact.getRootRelativePath());
    if (execPath.equals(resourcePath)) {
      builder.addPaths(":%s", resourcePath);
    } else {
      // execPath must end with resourcePath in all cases
      PathFragment rootPrefix = trimTail(execPath, resourcePath);
      builder.addPaths("%s:%s", rootPrefix, resourcePath);
    }
  }

  /**
   * Returns the root-part of a given path by trimming off the end specified by
   * a given tail. Assumes that the tail is known to match, and simply relies on
   * the segment lengths.
   */
  private static PathFragment trimTail(PathFragment path, PathFragment tail) {
    return path.subFragment(0, path.segmentCount() - tail.segmentCount());
  }

  /**
   * Builds the list of mappings between jars on the classpath and their
   * originating targets names.
   */
  private static ImmutableList<String> addJarsToTargets(
      NestedSet<Artifact> classpath, Collection<Artifact> directJars) {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (Artifact jar : classpath) {
      builder.add(directJars.contains(jar)
          ? "--direct_dependency"
          : "--indirect_dependency");
      builder.add(jar.getExecPathString());
      Label label = getTargetName(jar);
      builder.add(label.getPackageIdentifier().getRepository().isDefault()
          ? label.toString()
          : label.toPathFragment().toString());
    }
    return builder.build();
  }

  /**
   * Gets the name of the target that produced the given jar artifact.
   *
   * When specifying jars directly in the "srcs" attribute of a rule (mostly
   * for third_party libraries), there is no generating action, so we just
   * return the jar name in label form.
   */
  private static Label getTargetName(Artifact jar) {
    return Preconditions.checkNotNull(jar.getOwner(), jar);
  }

  /**
   * The actual command line executed for a compile action.
   */
  private static CommandLine spawnCommandLine(PathFragment javaExecutable, Artifact javaBuilderJar,
      Artifact langtoolsJar, ImmutableList<Artifact> instrumentationJars, Artifact paramFile,
      ImmutableList<String> javaBuilderJvmFlags, String javaBuilderMainClass,
      String pathDelimiter) {
    Preconditions.checkNotNull(langtoolsJar);
    Preconditions.checkNotNull(javaBuilderJar);

    CustomCommandLine.Builder builder =  CustomCommandLine.builder()
        .addPath(javaExecutable)
        // Langtools jar is placed on the boot classpath so that it can override classes
        // in the JRE. Typically this has no effect since langtools.jar does not have
        // classes in common with rt.jar. However, it is necessary when using a version
        // of javac.jar generated via ant from the langtools build.xml that is of a
        // different version than AND has an overlap in contents with the default
        // run-time (eg while upgrading the Java version).
        .addPaths("-Xbootclasspath/p:%s", langtoolsJar.getExecPath())
        .add(javaBuilderJvmFlags);
    if (!instrumentationJars.isEmpty()) {
      builder
          .addJoinExecPaths("-cp", pathDelimiter,
              Iterables.concat(instrumentationJars, ImmutableList.of(javaBuilderJar)))
          .add(javaBuilderMainClass);
    } else {
      // If there are no instrumentation jars, use the simpler '-jar' option to launch JavaBuilder.
      builder.addExecPath("-jar", javaBuilderJar);
    }
    return builder
        .addPaths("@%s", paramFile.getExecPath())
        .build();
  }

  /**
   * Builder class to construct Java compile actions.
   */
  public static class Builder {
    private final ActionOwner owner;
    private final AnalysisEnvironment analysisEnvironment;
    private final BuildConfiguration configuration;
    private final JavaSemantics semantics;

    private PathFragment javaExecutable;
    private List<Artifact> javabaseInputs = ImmutableList.of();
    private Artifact outputJar;
    private Artifact gensrcOutputJar;
    private Artifact outputDepsProto;
    private Artifact paramFile;
    private Artifact metadata;
    private final Collection<Artifact> sourceFiles = new ArrayList<>();
    private final Collection<Artifact> sourceJars = new ArrayList<>();
    private final Collection<Artifact> resources = new ArrayList<>();
    private final Collection<Artifact> classpathResources = new ArrayList<>();
    private final Collection<Artifact> translations = new LinkedHashSet<>();
    private BuildConfiguration.StrictDepsMode strictJavaDeps =
        BuildConfiguration.StrictDepsMode.OFF;
    private final Collection<Artifact> directJars = new ArrayList<>();
    private final Collection<Artifact> compileTimeDependencyArtifacts = new ArrayList<>();
    private List<String> javacOpts = new ArrayList<>();
    private boolean compressJar;
    private NestedSet<Artifact> classpathEntries =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private ImmutableList<Artifact> bootclasspathEntries = ImmutableList.of();
    private Artifact javaBuilderJar;
    private Artifact langtoolsJar;
    private ImmutableList<Artifact> instrumentationJars = ImmutableList.of();
    private PathFragment classDirectory;
    private PathFragment sourceGenDirectory;
    private PathFragment tempDirectory;
    private final List<Artifact> processorPath = new ArrayList<>();
    private final List<String> processorNames = new ArrayList<>();
    private String ruleKind;
    private Label targetLabel;

    /**
     * Creates a Builder from an owner and a build configuration.
     */
    public Builder(ActionOwner owner, AnalysisEnvironment analysisEnvironment,
        BuildConfiguration configuration, JavaSemantics semantics) {
      this.owner = owner;
      this.analysisEnvironment = analysisEnvironment;
      this.configuration = configuration;
      this.semantics = semantics;
    }

    /**
     * Creates a Builder from an owner and a build configuration.
     */
    public Builder(RuleContext ruleContext, JavaSemantics semantics) {
      this(ruleContext.getActionOwner(), ruleContext.getAnalysisEnvironment(),
          ruleContext.getConfiguration(), semantics);
    }

    public JavaCompileAction build() {
      // TODO(bazel-team): all the params should be calculated before getting here, and the various
      // aggregation code below should go away.
      List<String> jcopts = new ArrayList<>(javacOpts);
      JavaConfiguration javaConfiguration = configuration.getFragment(JavaConfiguration.class);
      if (!javaConfiguration.getJavaWarns().isEmpty()) {
        jcopts.add("-Xlint:" + Joiner.on(',').join(javaConfiguration.getJavaWarns()));
      }
      if (!bootclasspathEntries.isEmpty()) {
        jcopts.add("-bootclasspath");
        jcopts.add(
            Artifact.joinExecPaths(configuration.getHostPathSeparator(), bootclasspathEntries));
      }
      List<String> internedJcopts = new ArrayList<>();
      for (String jcopt : jcopts) {
        internedJcopts.add(StringCanonicalizer.intern(jcopt));
      }

      // Invariant: if strictJavaDeps is OFF, then directJars and
      // dependencyArtifacts are ignored
      if (strictJavaDeps == BuildConfiguration.StrictDepsMode.OFF) {
        directJars.clear();
        compileTimeDependencyArtifacts.clear();
      }

      // Invariant: if experimental_java_classpath is not set to 'javabuilder',
      // dependencyArtifacts are ignored
      if (javaConfiguration.getReduceJavaClasspath() != JavaClasspathMode.JAVABUILDER) {
        compileTimeDependencyArtifacts.clear();
      }

      if (paramFile == null) {
        paramFile = analysisEnvironment.getDerivedArtifact(
            ParameterFile.derivePath(outputJar.getRootRelativePath()),
            configuration.getBinDirectory());
      }

      // ImmutableIterable is safe to use here because we know that neither of the components of
      // the Iterable.concat() will change. Without ImmutableIterable, AbstractAction will
      // waste memory by making a preventive copy of the iterable.
      Iterable<Artifact> baseInputs = ImmutableIterable.from(Iterables.concat(
          javabaseInputs,
          bootclasspathEntries,
          ImmutableList.of(paramFile)));

      Preconditions.checkState(javaExecutable != null, owner);
      Preconditions.checkState(javaExecutable.isAbsolute() ^ !javabaseInputs.isEmpty(),
          javaExecutable);

      Collection<Artifact> outputs;
      ImmutableList.Builder<Artifact> outputsBuilder = ImmutableList.builder();
      outputsBuilder.add(outputJar);
      if (metadata != null) {
        outputsBuilder.add(metadata);
      }
      if (gensrcOutputJar != null) {
        outputsBuilder.add(gensrcOutputJar);
      }
      if (outputDepsProto != null) {
        outputsBuilder.add(outputDepsProto);
      }
      outputs = outputsBuilder.build();

      CustomCommandLine.Builder paramFileContentsBuilder = javaCompileCommandLine(
          semantics,
          configuration,
          classDirectory,
          sourceGenDirectory,
          tempDirectory,
          outputJar,
          gensrcOutputJar,
          compressJar,
          outputDepsProto,
          classpathEntries,
          processorPath,
          processorNames,
          translations,
          resources,
          classpathResources,
          sourceJars,
          sourceFiles,
          internedJcopts,
          directJars,
          strictJavaDeps,
          compileTimeDependencyArtifacts,
          ruleKind,
          targetLabel);
      semantics.buildJavaCommandLine(outputs, configuration, paramFileContentsBuilder);
      CommandLine paramFileContents = paramFileContentsBuilder.build();
      Action parameterFileWriteAction = new ParameterFileWriteAction(owner, paramFile,
          paramFileContents, ParameterFile.ParameterFileType.UNQUOTED, ISO_8859_1);
      analysisEnvironment.registerAction(parameterFileWriteAction);

      CommandLine javaBuilderCommandLine = spawnCommandLine(
          javaExecutable,
          javaBuilderJar,
          langtoolsJar,
          instrumentationJars,
          paramFile,
          javaConfiguration.getDefaultJavaBuilderJvmFlags(),
          semantics.getJavaBuilderMainClass(),
          configuration.getHostPathSeparator());

      return new JavaCompileAction(owner,
          baseInputs,
          outputs,
          paramFileContents,
          javaBuilderCommandLine,
          classDirectory,
          outputJar,
          classpathEntries,
          processorPath,
          langtoolsJar,
          javaBuilderJar,
          instrumentationJars,
          processorNames,
          translations,
          resources,
          classpathResources,
          sourceJars,
          sourceFiles,
          internedJcopts,
          directJars,
          strictJavaDeps,
          compileTimeDependencyArtifacts,

          semantics);
    }

    public Builder setParameterFile(Artifact paramFile) {
      this.paramFile = paramFile;
      return this;
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

    public Builder setGensrcOutputJar(Artifact gensrcOutputJar) {
      this.gensrcOutputJar = gensrcOutputJar;
      return this;
    }

    public Builder setOutputDepsProto(Artifact outputDepsProto) {
      this.outputDepsProto = outputDepsProto;
      return this;
    }

    public Builder setMetadata(Artifact metadata) {
      this.metadata = metadata;
      return this;
    }

    public Builder addSourceFile(Artifact sourceFile) {
      sourceFiles.add(sourceFile);
      return this;
    }

    public Builder addSourceFiles(Collection<Artifact> sourceFiles) {
      this.sourceFiles.addAll(sourceFiles);
      return this;
    }

    public Builder addSourceJars(Collection<Artifact> sourceJars) {
      this.sourceJars.addAll(sourceJars);
      return this;
    }

    public Builder addResources(Collection<Artifact> resources) {
      this.resources.addAll(resources);
      return this;
    }

    public Builder addClasspathResources(Collection<Artifact> classpathResources) {
      this.classpathResources.addAll(classpathResources);
      return this;
    }

    public Builder addTranslations(Collection<Artifact> translations) {
      this.translations.addAll(translations);
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

    /**
     * Accumulates the given jar artifacts as being provided by direct dependencies.
     */
    public Builder addDirectJars(Collection<Artifact> directJars) {
      this.directJars.addAll(directJars);
      return this;
    }

    public Builder addCompileTimeDependencyArtifacts(Collection<Artifact> dependencyArtifacts) {
      this.compileTimeDependencyArtifacts.addAll(dependencyArtifacts);
      return this;
    }

    public Builder setJavacOpts(Iterable<String> copts) {
      this.javacOpts = ImmutableList.copyOf(copts);
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

    public Builder setClassDirectory(PathFragment classDirectory) {
      this.classDirectory = classDirectory;
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

    public Builder addProcessorPaths(Collection<Artifact> processorPaths) {
      this.processorPath.addAll(processorPaths);
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

    public Builder setJavaBuilderJar(Artifact javaBuilderJar) {
      this.javaBuilderJar = javaBuilderJar;
      return this;
    }

    public Builder setInstrumentationJars(Iterable<Artifact> instrumentationJars) {
      this.instrumentationJars = ImmutableList.copyOf(instrumentationJars);
      return this;
    }

    public Builder setRuleKind(String ruleKind) {
      this.ruleKind = ruleKind;
      return this;
    }

    public Builder setTargetLabel(Label targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }
  }
}
