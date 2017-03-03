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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.devtools.build.lib.packages.Aspect.INJECTING_RULE_KIND_PARAMETER_KEY;
import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomMultiArgv;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.ImmutableIterable;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.proto.Deps;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Action that represents a Java compilation.
 */
@ThreadCompatible @Immutable
public final class JavaCompileAction extends AbstractAction {
  private static final String JACOCO_INSTRUMENTATION_PROCESSOR = "jacoco";

  private static final String GUID = "786e174d-ed97-4e79-9f61-ae74430714cf";

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpuIo(750 /*MB*/, 0.5 /*CPU*/, 0.0 /*IO*/);

  /** Default number of unused jars below which we ignore minimum classpath optimization. */
  private static final int MINIMUM_REDUCTION_TO_SAVE_USED_INPUTS = 0;

  /** A {@link clientEnvironmentVariables} entry that sets the UTF-8 charset. */
  static final ImmutableMap<String, String> UTF8_ENVIRONMENT =
      ImmutableMap.of("LC_CTYPE", "en_US.UTF-8");

  private static final Logger logger = Logger.getLogger(JavaCompileAction.class.getName());

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

  /** The list of bootclasspath entries to specify to javac. */
  private final ImmutableList<Artifact> bootclasspathEntries;

  /**
   * The path to the extdir to specify to javac.
   */
  private final ImmutableList<Artifact> extdirInputs;

  /**
   * The list of classpath entries to search for annotation processors.
   */
  private final ImmutableList<Artifact> processorPath;

  /**
   * The list of annotation processor classes to run.
   */
  private final ImmutableList<String> processorNames;

  /**
   * The set of resources to put into the jar.
   */
  private final ImmutableList<Artifact> resources;

  /**
   * The set of resource jars to merge into the jar.
   */
  private final NestedSet<Artifact> resourceJars;

  /** The number of resources that will be put into the jar. */
  private final int resourceCount;

  /** Set of additional Java source files to compile. */
  private final ImmutableList<Artifact> sourceJars;

  /**
   * The set of explicit Java source files to compile.
   */
  private final ImmutableList<Artifact> sourceFiles;

  /**
   * The compiler options to pass to javac.
   */
  private final ImmutableList<String> javacOpts;

  /** The subset of classpath jars provided by direct dependencies. */
  private final NestedSet<Artifact> directJars;

  /** The ExecutionInfo to be used when creating the SpawnAction for this compilation. */
  private final ImmutableMap<String, String> executionInfo;

  /**
   * The level of strict dependency checks (off, warnings, or errors).
   */
  private final BuildConfiguration.StrictDepsMode strictJavaDeps;

  /**
   * The set of .jdeps artifacts provided by direct dependencies.
   */
  private final ImmutableList<Artifact> compileTimeDependencyArtifacts;

  /** Saved copy of the baseInputs needed to reconstruct the full inputs. */
  private final Iterable<Artifact> baseInputs;

  /** If non-null, use this command line base for calling JavaBuilder with minimum classpath. */
  private final CommandLine minimumCommandLineBase;

  private final String pathDelimiter;

  /** Inputs that were actually used for the previous compilation, if successful. */
  private Iterable<Artifact> usedInputs;

  /** Actual, complete command line for minimum compile optimization. */
  private CommandLine minCommandLine;

  /**
   * Constructs an action to compile a set of Java source files to class files.
   *
   * @param owner the action owner, typically a java_* RuleConfiguredTarget.
   * @param baseInputs the set of the input artifacts of the compile action without the parameter
   *     file action;
   * @param outputs the outputs of the action
   * @param paramFile the file containing the command line arguments to JavaBuilder
   * @param javaCompileCommandLine the command line for the java library builder - it's actually
   *     written to the parameter file, but other parts (for example, ide_build_info) need access to
   *     the data
   * @param commandLine the actual invocation command line
   * @param minimumCommandLineBase minimum classpath invocation command line, without inputs
   * @param resourceCount the count of all resource inputs
   */
  private JavaCompileAction(
      ActionOwner owner,
      NestedSet<Artifact> tools,
      Iterable<Artifact> baseInputs,
      Collection<Artifact> outputs,
      Artifact paramFile,
      CommandLine javaCompileCommandLine,
      CommandLine commandLine,
      PathFragment classDirectory,
      Artifact outputJar,
      NestedSet<Artifact> classpathEntries,
      ImmutableList<Artifact> bootclasspathEntries,
      ImmutableList<Artifact> extdirInputs,
      List<Artifact> processorPath,
      List<String> processorNames,
      Map<PathFragment, Artifact> resources,
      NestedSet<Artifact> resourceJars,
      Collection<Artifact> sourceJars,
      Collection<Artifact> sourceFiles,
      List<String> javacOpts,
      NestedSet<Artifact> directJars,
      Map<String, String> executionInfo,
      BuildConfiguration.StrictDepsMode strictJavaDeps,
      Collection<Artifact> compileTimeDependencyArtifacts,
      CommandLine minimumCommandLineBase,
      String pathDelimiter,
      int resourceCount) {
    super(
        owner,
        tools,
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(classpathEntries)
            .addAll(compileTimeDependencyArtifacts)
            .addAll(baseInputs)
            .add(paramFile)
            .addTransitive(tools)
            .build(),
        outputs);
    this.javaCompileCommandLine = javaCompileCommandLine;
    this.commandLine = commandLine;

    this.classDirectory = checkNotNull(classDirectory);
    this.outputJar = outputJar;
    this.classpathEntries = classpathEntries;
    this.bootclasspathEntries = ImmutableList.copyOf(bootclasspathEntries);
    this.extdirInputs = extdirInputs;
    this.processorPath = ImmutableList.copyOf(processorPath);
    this.processorNames = ImmutableList.copyOf(processorNames);
    this.resources = ImmutableList.copyOf(resources.values());
    this.resourceJars = resourceJars;
    this.sourceJars = ImmutableList.copyOf(sourceJars);
    this.sourceFiles = ImmutableList.copyOf(sourceFiles);
    this.javacOpts = ImmutableList.copyOf(javacOpts);
    this.directJars = checkNotNull(directJars, "directJars must not be null");
    this.executionInfo = ImmutableMap.copyOf(executionInfo);
    this.strictJavaDeps = strictJavaDeps;
    this.compileTimeDependencyArtifacts = ImmutableList.copyOf(compileTimeDependencyArtifacts);
    this.baseInputs = ImmutableList.copyOf(baseInputs);
    this.minimumCommandLineBase = minimumCommandLineBase;
    this.pathDelimiter = pathDelimiter;
    this.resourceCount = resourceCount;
  }

  /**
   * Returns the given (passed to constructor) source files.
   */
  @VisibleForTesting
  Collection<Artifact> getSourceFiles() {
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

  @VisibleForTesting
  public NestedSet<Artifact> getResourceJars() {
    return resourceJars;
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
  public ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @VisibleForTesting
  public NestedSet<Artifact> getDirectJars() {
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

  /** Returns the command and arguments for a java compile action. */
  public List<String> getCommand() {
    // If available, use the saved minCommandLine, otherwise use the command line with full inputs.
    return ImmutableList.copyOf(
        (minCommandLine != null ? minCommandLine : commandLine).arguments());
  }

  @VisibleForTesting
  Spawn createSpawn() {
    return new BaseSpawn(getCommand(), UTF8_ENVIRONMENT, executionInfo, this, LOCAL_RESOURCES) {
      @Override
      public Iterable<? extends ActionInput> getInputFiles() {
        // Reduce inputs for minclasspath compile. Requires use of minCommandLine.
        return usedInputs != null ? usedInputs : super.getInputFiles();
      }
    };
  }

  @Override
  @ThreadCompatible
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    boolean executeFullCompile = true;
    if (usedInputs != null) {
      try {
        getContext(executor).exec(createSpawn(), actionExecutionContext);
        executeFullCompile = false;
      } catch (ExecException e) {
        logger.info("Minimum classpath failed for " + getOwner().getLabel().toShorthandString());
        minCommandLine = null;
        usedInputs = null;
      }
    }
    if (executeFullCompile) {
      try {
        getContext(executor).exec(createSpawn(), actionExecutionContext);
      } catch (ExecException e) {
        throw e.toActionExecutionException(
            "Java compilation in rule '" + getOwner().getLabel() + "'",
            executor.getVerboseFailures(),
            this);
      }
    }
    maybeSaveUsedInputs();
  }
  /**
   * If enabled, read and save the contents of the output '.jdeps' file for minimum classpath
   * incremental compile on the next call to execute.
   */
  @VisibleForTesting
  void maybeSaveUsedInputs() {
    minCommandLine = null;
    usedInputs = null;
    if (minimumCommandLineBase != null) {
      // Find the actually needed dependencies from the just-generated jdeps file.
      // TODO(b/30902566): Cache jdeps md5, avoid rereading and recreating command when unchanged.
      Set<String> usedInputJars = readJdeps(outputJar);
      if (usedInputJars == null) {
        return;
      }
      List<Artifact> orderedClasspath = classpathEntries.toList();
      // Only continue if we anticipate the added work and memory will pay off.
      if (orderedClasspath.size() - usedInputJars.size() <= MINIMUM_REDUCTION_TO_SAVE_USED_INPUTS) {
        return;
      }
      ImmutableList.Builder<Artifact> minInputsBuilder = ImmutableList.builder();
      for (Artifact artifact : orderedClasspath) {
        if (usedInputJars.contains(artifact.getExecPathString())) {
          minInputsBuilder.add(artifact);
        }
      }
      final ImmutableList<Artifact> minimumInputs = minInputsBuilder.build();
      // The two things needed to enable minimum incremental classpath compile - command & inputs
      CustomCommandLine.Builder minCommandLineBuilder = CustomCommandLine.builder();
      minCommandLineBuilder.add(minimumCommandLineBase.arguments());
      if (!minimumInputs.isEmpty()) {
        minCommandLineBuilder.addJoinExecPaths("--classpath", pathDelimiter, minimumInputs);
      }
      minCommandLineBuilder
          .add("--strict_java_deps")
          .add(strictJavaDeps.toString())
          .add(new JarsToTargetsArgv(minimumInputs, directJars));
      minCommandLine = minCommandLineBuilder.build();

      // Keep in sync with inputs in constructor call to 'super', except do not
      // include compileTimeDependencyArtifacts or paramFile, which are unneeded here.
      usedInputs = Iterables.concat(minimumInputs, baseInputs, getTools());
    }
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addBoolean(minimumCommandLineBase != null);
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
    StringBuilder sb = new StringBuilder("Building ");
    sb.append(outputJar.prettyPrint());
    sb.append(" (");
    boolean first = true;
    first = appendCount(sb, first, sourceFiles.size(), "source file");
    first = appendCount(sb, first, sourceJars.size(), "source jar");
    first = appendCount(sb, first, resourceCount, "resource");
    sb.append(")");
    return sb.toString();
  }

  /**
   * Append an input count to the progress message, e.g. "2 source jars". If an input
   * count has already been appended, prefix with ", ".
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
    info.addAllBootclasspath(Artifact.toExecPaths(getBootclasspath()));
    info.addAllSourcepath(Artifact.toExecPaths(getSourceJars()));
    info.addAllJavacOpt(getJavacOpts());
    info.addAllProcessor(getProcessorNames());
    info.addAllProcessorpath(Artifact.toExecPaths(getProcessorpath()));
    info.setOutputjar(getOutputJar().getExecPathString());

    return super.getExtraActionInfo()
        .setExtension(JavaCompileInfo.javaCompileInfo, info.build());
  }

  /**
   * Collect common command line arguments together in a single ArgvFragment.
   *
   * @param classDirectory the directory in which generated classfiles are placed relative to the
   *     exec root
   * @param sourceGenDirectory the directory where source files generated by annotation processors
   *     should be stored.
   * @param tempDirectory a directory in which the library builder can store temporary files
   *     relative to the exec root
   * @param outputJar output jar
   * @param compressJar if true compress the output jar
   * @param outputDepsProto the proto file capturing dependency information
   * @param processorPath the classpath files where javac should search for annotation processors
   * @param processorPathDirs the classpath dirs where javac should search for annotation processors
   * @param processorNames the classes that javac should use as annotation processors
   * @param messages the message files for translation
   * @param resources the set of resources to put into the jar
   * @param classpathResources the set of classpath resources to put into the jar
   * @param sourceJars the set of jars containing additional source files to compile
   * @param sourceFiles the set of explicit Java source files to compile
   * @param javacOpts the compiler options to pass to javac
   */
  private static CustomMultiArgv commonJavaBuilderArgs(
      final JavaSemantics semantics,
      final PathFragment classDirectory,
      final PathFragment sourceGenDirectory,
      final PathFragment tempDirectory,
      final Artifact outputJar,
      final Artifact gensrcOutputJar,
      final Artifact manifestProto,
      final boolean compressJar,
      final Artifact outputDepsProto,
      final List<Artifact> processorPath,
      final Set<PathFragment> processorPathDirs,
      final List<String> processorNames,
      final Collection<Artifact> messages,
      final Map<PathFragment, Artifact> resources,
      final NestedSet<Artifact> resourceJars,
      final Collection<Artifact> classpathResources,
      final Collection<Artifact> sourceJars,
      final Collection<Artifact> sourceFiles,
      final Collection<Artifact> extdirInputs,
      final Collection<Artifact> bootclasspathEntries,
      final List<String> javacOpts,
      final String ruleKind,
      final Label targetLabel,
      final String pathSeparator) {
    return new CustomMultiArgv() {
      @Override
      public Iterable<String> argv() {
        checkNotNull(classDirectory);
        checkNotNull(tempDirectory);
        CustomCommandLine.Builder result = CustomCommandLine.builder();

        result.add("--classdir").addPath(classDirectory);
        result.add("--tempdir").addPath(tempDirectory);
        if (outputJar != null) {
          result.addExecPath("--output", outputJar);
        }
        if (sourceGenDirectory != null) {
          result.add("--sourcegendir").addPath(sourceGenDirectory);
        }
        if (gensrcOutputJar != null) {
          result.addExecPath("--generated_sources_output", gensrcOutputJar);
        }
        if (manifestProto != null) {
          result.addExecPath("--output_manifest_proto", manifestProto);
        }
        if (compressJar) {
          result.add("--compress_jar");
        }
        if (outputDepsProto != null) {
          result.addExecPath("--output_deps_proto", outputDepsProto);
        }
        if (!extdirInputs.isEmpty()) {
          result.addJoinExecPaths("--extdir", pathSeparator, extdirInputs);
        }
        if (!bootclasspathEntries.isEmpty()) {
          result.addJoinExecPaths("--bootclasspath", pathSeparator, bootclasspathEntries);
        }
        if (!processorPath.isEmpty() || !processorPathDirs.isEmpty()) {
          ImmutableList.Builder<String> execPathStrings = ImmutableList.<String>builder();
          execPathStrings.addAll(Artifact.toExecPaths(processorPath));
          for (PathFragment processorPathDir : processorPathDirs) {
            execPathStrings.add(processorPathDir.toString());
          }
          result.addJoinStrings("--processorpath", pathSeparator, execPathStrings.build());
        }
        if (!processorNames.isEmpty()) {
          result.add("--processors", processorNames);
        }
        if (!messages.isEmpty()) {
          result.add("--messages");
          for (Artifact message : messages) {
            addAsResourcePrefixedExecPath(
                semantics.getDefaultJavaResourcePath(message.getRootRelativePath()),
                message,
                result);
          }
        }
        if (!resources.isEmpty()) {
          result.add("--resources");
          for (Map.Entry<PathFragment, Artifact> resource : resources.entrySet()) {
            addAsResourcePrefixedExecPath(resource.getKey(), resource.getValue(), result);
          }
        }
        if (!resourceJars.isEmpty()) {
          result.addExecPaths("--resource_jars", resourceJars);
        }
        if (!classpathResources.isEmpty()) {
          result.addExecPaths("--classpath_resources", classpathResources);
        }
        if (!sourceJars.isEmpty()) {
          result.addExecPaths("--source_jars", sourceJars);
        }
        if (!sourceFiles.isEmpty()) {
          result.addExecPaths("--sources", sourceFiles);
        }
        if (!javacOpts.isEmpty()) {
          result.add("--javacopts", javacOpts);
        }
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
            // @-prefixed strings will be assumed to be filenames and expanded by
            // {@link JavaLibraryBuildRequest}, so add an extra &at; to escape it.
            result.add("@" + targetLabel);
          }
        }
        return result.build().arguments();
      }
    };
  }

  /**
   * Creates an instance.
   * @param commonJavaBuilderArgs common flag values consumed by JavaBuilder
   * @param configuration the build configuration, which provides the default options and the path
   *     to the compiler, etc.
   * @param classpath the complete classpath, the directory in which generated classfiles are placed
   */
  private static CustomCommandLine.Builder javaCompileCommandLine(
      CustomMultiArgv commonJavaBuilderArgs,
      final BuildConfiguration configuration,
      final NestedSet<Artifact> classpath,
      final NestedSet<Artifact> directJars,
      BuildConfiguration.StrictDepsMode strictJavaDeps,
      Collection<Artifact> compileTimeDependencyArtifacts) {
    CustomCommandLine.Builder result = CustomCommandLine.builder();

    result.add(commonJavaBuilderArgs);
    if (!classpath.isEmpty()) {
      result.addJoinExecPaths("--classpath", configuration.getHostPathSeparator(), classpath);
    }

    // strict_java_deps controls whether the mapping from jars to targets is
    // written out and whether we try to minimize the compile-time classpath.
    if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
      result.add("--strict_java_deps");
      result.add(strictJavaDeps.toString());
      result.add(new JarsToTargetsArgv(classpath, directJars));

      if (configuration.getFragment(JavaConfiguration.class).getReduceJavaClasspath()
          != JavaClasspathMode.OFF) {
        result.add("--reduce_classpath");

        if (!compileTimeDependencyArtifacts.isEmpty()) {
          result.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
        }
      }
    }
    return result;
  }

  private static void addAsResourcePrefixedExecPath(PathFragment resourcePath,
      Artifact artifact, CustomCommandLine.Builder builder) {
    PathFragment execPath = artifact.getExecPath();
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
   * Builds the list of mappings between jars on the classpath and their originating targets names.
   */
  @VisibleForTesting
  static class JarsToTargetsArgv extends CustomMultiArgv {
    private final Iterable<Artifact> classpath;
    private final NestedSet<Artifact> directJars;

    @VisibleForTesting
    JarsToTargetsArgv(Iterable<Artifact> classpath, NestedSet<Artifact> directJars) {
      this.classpath = classpath;
      this.directJars = directJars;
    }

    @Override
    public Iterable<String> argv() {
      Set<Artifact> directJarSet = directJars.toSet();
      ImmutableList.Builder<String> builder = ImmutableList.builder();
      for (Artifact jar : classpath) {
        builder.add(directJarSet.contains(jar) ? "--direct_dependency" : "--indirect_dependency");
        builder.add(jar.getExecPathString());
        builder.add(getArtifactOwnerGeneralizedLabel(jar));
      }
      return builder.build();
    }

    private String getArtifactOwnerGeneralizedLabel(Artifact artifact) {
      ArtifactOwner owner = checkNotNull(artifact.getArtifactOwner(), artifact);
      StringBuilder result = new StringBuilder();
      Label label = owner.getLabel();
      result.append(
          label.getPackageIdentifier().getRepository().isDefault()
                  || label.getPackageIdentifier().getRepository().isMain()
              ? label.toString()
              // Escape '@' prefix for .params file.
              : "@" + label);

      if (owner instanceof AspectValue.AspectKey) {
        AspectValue.AspectKey aspectOwner = (AspectValue.AspectKey) owner;
        ImmutableCollection<String> injectingRuleKind =
            aspectOwner.getParameters().getAttribute(INJECTING_RULE_KIND_PARAMETER_KEY);
        if (injectingRuleKind.size() == 1) {
          result.append(' ').append(getOnlyElement(injectingRuleKind));
        }
      }

      return result.toString();
    }
  }

  /** Creates an ArgvFragment containing the common initial command line arguments */
  private static CustomMultiArgv spawnCommandLineBase(
      final PathFragment javaExecutable,
      final Artifact javaBuilderJar,
      final Artifact langtoolsJar,
      final ImmutableList<Artifact> instrumentationJars,
      final ImmutableList<String> javaBuilderJvmFlags,
      final String javaBuilderMainClass,
      final String pathDelimiter) {
    return new CustomMultiArgv() {
      @Override
      public Iterable<String> argv() {
        checkNotNull(langtoolsJar);
        checkNotNull(javaBuilderJar);

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
          // If there are no instrumentation jars, use simpler '-jar' option to launch JavaBuilder.
          builder.addExecPath("-jar", javaBuilderJar);
        }
        return builder.build().arguments();
      }
    };
  }

  @VisibleForTesting
  static Set<String> readJdeps(Artifact outputJar) {
    Set<String> jdeps = new HashSet<>();
    Path jdepsFile = FileSystemUtils.replaceExtension(outputJar.getPath(), ".jdeps");
    String label = outputJar.getOwnerLabel().toShorthandString();
    if (!jdepsFile.exists() || !jdepsFile.isFile()) {
      logger.warning("Jdeps file missing for " + label);
      return null;
    }
    try (InputStream bis = new BufferedInputStream(jdepsFile.getInputStream())) {
      Deps.Dependencies deps = Deps.Dependencies.parseFrom(bis);
      if (!deps.hasSuccess() || !deps.getSuccess() || !deps.hasRuleLabel()) {
        logger.warning("Cannot use jdeps file for " + label);
        return null;
      }
      for (Deps.Dependency dep : deps.getDependencyList()) {
        jdeps.add(dep.getPath());
      }
    } catch (IOException e) {
      logger.warning("Failed to read jdeps file for " + label);
      return null;
    }
    return jdeps;
  }

  /**
   * Tells {@link Builder} how to create new artifacts. Is there so that {@link Builder} can be
   * exercised in tests without creating a full {@link RuleContext}.
   */
  public interface ArtifactFactory {

    /**
     * Create an artifact with the specified root-relative path under the specified root.
     */
    Artifact create(PathFragment rootRelativePath, Root root);
  }

  @VisibleForTesting
  static ArtifactFactory createArtifactFactory(final AnalysisEnvironment env) {
    return new ArtifactFactory() {
      @Override
      public Artifact create(PathFragment rootRelativePath, Root root) {
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
    private Artifact gensrcOutputJar;
    private Artifact manifestProtoOutput;
    private Artifact outputDepsProto;
    private Collection<Artifact> additionalOutputs;
    private Artifact paramFile;
    private Artifact metadata;
    private final Collection<Artifact> sourceFiles = new ArrayList<>();
    private final Collection<Artifact> sourceJars = new ArrayList<>();
    private final Map<PathFragment, Artifact> resources = new LinkedHashMap<>();
    private final NestedSetBuilder<Artifact> resourceJars = NestedSetBuilder.stableOrder();
    private final Collection<Artifact> classpathResources = new ArrayList<>();
    private final Collection<Artifact> translations = new LinkedHashSet<>();
    private BuildConfiguration.StrictDepsMode strictJavaDeps =
        BuildConfiguration.StrictDepsMode.OFF;
    private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private final Collection<Artifact> compileTimeDependencyArtifacts = new ArrayList<>();
    private List<String> javacOpts = new ArrayList<>();
    private ImmutableList<String> javacJvmOpts = ImmutableList.of();
    private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
    private boolean compressJar;
    private NestedSet<Artifact> classpathEntries =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private ImmutableList<Artifact> bootclasspathEntries = ImmutableList.of();
    private ImmutableList<Artifact> extdirInputs = ImmutableList.of();
    private Artifact javaBuilderJar;
    private Artifact langtoolsJar;
    private ImmutableList<Artifact> instrumentationJars = ImmutableList.of();
    private PathFragment sourceGenDirectory;
    private PathFragment tempDirectory;
    private PathFragment classDirectory;
    private final List<Artifact> processorPath = new ArrayList<>();
    private final Set<PathFragment> processorPathDirs = new LinkedHashSet<>();
    private final List<String> processorNames = new ArrayList<>();
    private String ruleKind;
    private Label targetLabel;

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
      this(ruleContext.getActionOwner(),
          ruleContext.getAnalysisEnvironment(),
          new ArtifactFactory() {
            @Override
            public Artifact create(PathFragment rootRelativePath, Root root) {
              return ruleContext.getDerivedArtifact(rootRelativePath, root);
            }
          },
          ruleContext.getConfiguration(), semantics);
    }

    /**
     * May add extra command line options to the Java compile command line.
     */
    private static void buildJavaCommandLine(
        Collection<Artifact> outputs,
        BuildConfiguration configuration,
        CustomCommandLine.Builder result,
        Label targetLabel) {
      Artifact metadata = null;
      for (Artifact artifact : outputs) {
        if (artifact.getExecPathString().endsWith(".em")) {
          metadata = artifact;
          break;
        }
      }

      if (metadata == null) {
        return;
      }

      result.add("--post_processor");
      result.addExecPath(JACOCO_INSTRUMENTATION_PROCESSOR, metadata);
      result.addPath(
          configuration
              .getCoverageMetadataDirectory(targetLabel.getPackageIdentifier().getRepository())
              .getExecPath());
      result.add("-*Test");
      result.add("-*TestCase");
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
        compileTimeDependencyArtifacts.clear();
      }

      // Invariant: if java_classpath is set to 'off', dependencyArtifacts are ignored
      JavaConfiguration javaConfiguration = configuration.getFragment(JavaConfiguration.class);
      if (javaConfiguration.getReduceJavaClasspath() == JavaClasspathMode.OFF) {
        compileTimeDependencyArtifacts.clear();
      }

      if (paramFile == null) {
        paramFile = artifactFactory.create(
            ParameterFile.derivePath(outputJar.getRootRelativePath()),
            configuration.getBinDirectory(targetLabel.getPackageIdentifier().getRepository()));
      }

      NestedSet<Artifact> resourceJars = this.resourceJars.build();

      // ImmutableIterable is safe to use here because we know that none of the components of
      // the Iterable.concat() will change. Without ImmutableIterable, AbstractAction will
      // waste memory by making a preventive copy of the iterable.
      Iterable<Artifact> baseInputs = ImmutableIterable.from(Iterables.concat(
          processorPath,
          translations,
          resources.values(),
          resourceJars,
          sourceJars,
          sourceFiles,
          classpathResources,
          javabaseInputs,
          bootclasspathEntries,
          extdirInputs));

      Preconditions.checkState(javaExecutable != null, owner);
      Preconditions.checkState(javaExecutable.isAbsolute() ^ !javabaseInputs.isEmpty(),
          javaExecutable);

      ImmutableList.Builder<Artifact> outputsBuilder = ImmutableList.<Artifact>builder()
          .addAll(
              new ArrayList<>(Collections2.filter(Arrays.asList(
                  outputJar,
                  metadata,
                  gensrcOutputJar,
                  manifestProtoOutput,
                  outputDepsProto), Predicates.notNull())));
      if (additionalOutputs != null) {
        outputsBuilder.addAll(additionalOutputs);
      }
      ImmutableList<Artifact> outputs = outputsBuilder.build();

      CustomMultiArgv commonJavaBuilderArgs =
          commonJavaBuilderArgs(
              semantics,
              classDirectory,
              sourceGenDirectory,
              tempDirectory,
              outputJar,
              gensrcOutputJar,
              manifestProtoOutput,
              compressJar,
              outputDepsProto,
              processorPath,
              processorPathDirs,
              processorNames,
              translations,
              resources,
              resourceJars,
              classpathResources,
              sourceJars,
              sourceFiles,
              extdirInputs,
              bootclasspathEntries,
              internedJcopts,
              ruleKind,
              targetLabel,
              pathSeparator);

      CustomCommandLine.Builder paramFileContentsBuilder = javaCompileCommandLine(
          commonJavaBuilderArgs,
          configuration,
          classpathEntries,
          directJars,
          strictJavaDeps,
          compileTimeDependencyArtifacts
      );
      buildJavaCommandLine(
          outputs, configuration, paramFileContentsBuilder, targetLabel);
      CommandLine paramFileContents = paramFileContentsBuilder.build();
      Action parameterFileWriteAction = new ParameterFileWriteAction(owner, paramFile,
          paramFileContents, ParameterFile.ParameterFileType.UNQUOTED, ISO_8859_1);
      analysisEnvironment.registerAction(parameterFileWriteAction);

      CustomMultiArgv spawnCommandLineBase = spawnCommandLineBase(
          javaExecutable,
          javaBuilderJar,
          langtoolsJar,
          instrumentationJars,
          javacJvmOpts,
          semantics.getJavaBuilderMainClass(),
          pathSeparator);

      // The actual params-file-based command line executed for a compile action.
      CommandLine javaBuilderCommandLine = CustomCommandLine.builder()
          .add(spawnCommandLineBase)
          .addPaths("@%s", paramFile.getExecPath())
          .build();

      NestedSet<Artifact> tools =
          NestedSetBuilder.<Artifact>stableOrder()
              .add(langtoolsJar)
              .add(javaBuilderJar)
              .addAll(instrumentationJars)
              .build();

      // Minimum compile command line null unless explicitly enabled via JavaClasspathMode.
      CommandLine minimumCommandLineBase = null;
      if (javaConfiguration.getReduceJavaClasspath() == JavaClasspathMode.EXPERIMENTAL_BLAZE) {
        minimumCommandLineBase =
            CustomCommandLine.builder()
                .add(spawnCommandLineBase)
                .add(commonJavaBuilderArgs)
                .build();
      }

      return new JavaCompileAction(
          owner,
          tools,
          baseInputs,
          outputs,
          paramFile,
          paramFileContents,
          javaBuilderCommandLine,
          classDirectory,
          outputJar,
          classpathEntries,
          bootclasspathEntries,
          extdirInputs,
          processorPath,
          processorNames,
          resources,
          resourceJars,
          sourceJars,
          sourceFiles,
          internedJcopts,
          directJars,
          executionInfo,
          strictJavaDeps,
          compileTimeDependencyArtifacts,
          minimumCommandLineBase,
          pathSeparator,
          resources.size() + classpathResources.size() + translations.size());
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

    public Builder addResources(Map<PathFragment, Artifact> resources) {
      this.resources.putAll(resources);
      return this;
    }

    public Builder addResourceJars(NestedSet<Artifact> resourceJars) {
      this.resourceJars.addTransitive(resourceJars);
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

    /** Accumulates the given jar artifacts as being provided by direct dependencies. */
    public Builder setDirectJars(NestedSet<Artifact> directJars) {
      this.directJars = checkNotNull(directJars, "directJars must not be null");
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

    public Builder addProcessorPaths(Collection<Artifact> processorPaths) {
      this.processorPath.addAll(processorPaths);
      return this;
    }

    public Builder addProcessorPathDirs(Collection<PathFragment> processorPathDirs) {
      this.processorPathDirs.addAll(processorPathDirs);
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
