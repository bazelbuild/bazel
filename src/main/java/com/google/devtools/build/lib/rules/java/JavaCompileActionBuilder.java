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
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.actions.extra.JavaCompileInfo;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction.ExtraActionInfoSupplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider.JavaPluginInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Java compilation action builder. */
public final class JavaCompileActionBuilder {

  private static final String JACOCO_INSTRUMENTATION_PROCESSOR = "jacoco";

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpuIo(750 /*MB*/, 0.5 /*CPU*/, 0.0 /*IO*/);

  /** Environment variable that sets the UTF-8 charset. */
  static final ImmutableMap<String, String> UTF8_ENVIRONMENT =
      ImmutableMap.of("LC_CTYPE", "en_US.UTF-8");

  static final String MNEMONIC = "Javac";

  /** Returns true if this is a Java compile action. */
  public static boolean isJavaCompileAction(ActionAnalysisMetadata action) {
    return action != null && action.getMnemonic().equals(MNEMONIC);
  }

  @ThreadCompatible
  @Immutable
  @AutoCodec
  static class JavaCompileExtraActionInfoSupplier implements ExtraActionInfoSupplier {

    private final Artifact outputJar;

    /** The list of classpath entries to specify to javac. */
    private final NestedSet<Artifact> classpathEntries;

    /** The list of bootclasspath entries to specify to javac. */
    private final ImmutableList<Artifact> bootclasspathEntries;

    /** The list of classpath entries to search for annotation processors. */
    private final NestedSet<Artifact> processorPath;

    /** The list of annotation processor classes to run. */
    private final NestedSet<String> processorNames;

    /** Set of additional Java source files to compile. */
    private final ImmutableList<Artifact> sourceJars;

    /** The set of explicit Java source files to compile. */
    private final ImmutableSet<Artifact> sourceFiles;

    /** The compiler options to pass to javac. */
    private final ImmutableList<String> javacOpts;

    private CommandLine commandLine;

    JavaCompileExtraActionInfoSupplier(
        Artifact outputJar,
        NestedSet<Artifact> classpathEntries,
        ImmutableList<Artifact> bootclasspathEntries,
        NestedSet<Artifact> processorPath,
        NestedSet<String> processorNames,
        ImmutableList<Artifact> sourceJars,
        ImmutableSet<Artifact> sourceFiles,
        ImmutableList<String> javacOpts,
        CommandLine commandLine) {
      this.outputJar = outputJar;
      this.classpathEntries = classpathEntries;
      this.bootclasspathEntries = bootclasspathEntries;
      this.processorPath = processorPath;
      this.processorNames = processorNames;
      this.sourceJars = sourceJars;
      this.sourceFiles = sourceFiles;
      this.javacOpts = javacOpts;
      this.commandLine = commandLine;
    }

    @Override
    public void extend(ExtraActionInfo.Builder builder) {
      JavaCompileInfo.Builder info = JavaCompileInfo.newBuilder();
      info.addAllSourceFile(Artifact.toExecPaths(sourceFiles));
      info.addAllClasspath(Artifact.toExecPaths(classpathEntries));
      info.addAllBootclasspath(Artifact.toExecPaths(bootclasspathEntries));
      info.addAllSourcepath(Artifact.toExecPaths(sourceJars));
      info.addAllJavacOpt(javacOpts);
      info.addAllProcessor(processorNames);
      info.addAllProcessorpath(Artifact.toExecPaths(processorPath));
      info.setOutputjar(outputJar.getExecPathString());
      try {
        info.addAllArgument(commandLine.arguments());
      } catch (CommandLineExpansionException e) {
        throw new AssertionError("JavaCompileAction command line expansion cannot fail", e);
      }
      builder.setExtension(JavaCompileInfo.javaCompileInfo, info.build());
    }
  }

  private PathFragment javaExecutable;
  private List<Artifact> javabaseInputs = ImmutableList.of();
  private Artifact outputJar;
  private Artifact nativeHeaderOutput;
  private Artifact gensrcOutputJar;
  private Artifact manifestProtoOutput;
  private Artifact outputDepsProto;
  private Collection<Artifact> additionalOutputs;
  private Artifact metadata;
  private Artifact artifactForExperimentalCoverage;
  private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
  private ImmutableList<Artifact> sourceJars = ImmutableList.of();
  private StrictDepsMode strictJavaDeps = StrictDepsMode.ERROR;
  private String fixDepsTool = "add_dep";
  private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private NestedSet<Artifact> compileTimeDependencyArtifacts =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  private List<String> javacOpts = new ArrayList<>();
  private ImmutableList<String> javacJvmOpts = ImmutableList.of();
  private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
  private boolean compressJar;
  private NestedSet<Artifact> classpathEntries = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
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
  private JavaPluginInfo plugins = JavaPluginInfo.empty();
  private NestedSet<Artifact> extraData = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
  private Label targetLabel;
  @Nullable private String injectingRuleKind;

  public void build(RuleContext ruleContext, JavaSemantics javaSemantics) {
    // TODO(bazel-team): all the params should be calculated before getting here, and the various
    // aggregation code below should go away.
    ImmutableList<String> internedJcopts =
        javacOpts.stream().map(StringCanonicalizer::intern).collect(toImmutableList());

    // Invariant: if strictJavaDeps is OFF, then directJars and
    // dependencyArtifacts are ignored
    if (strictJavaDeps == StrictDepsMode.OFF) {
      directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
      compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    // Invariant: if java_classpath is set to 'off', dependencyArtifacts are ignored
    JavaConfiguration javaConfiguration =
        ruleContext.getConfiguration().getFragment(JavaConfiguration.class);
    if (javaConfiguration.getReduceJavaClasspath() == JavaClasspathMode.OFF) {
      compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }

    Preconditions.checkState(javaExecutable != null, ruleContext.getActionOwner());

    SpawnAction.Builder builder = new SpawnAction.Builder();

    builder.addOutput(outputJar);
    Stream.of(metadata, gensrcOutputJar, manifestProtoOutput, outputDepsProto, nativeHeaderOutput)
        .filter(x -> x != null)
        .forEachOrdered(builder::addOutput);
    if (additionalOutputs != null) {
      builder.addOutputs(additionalOutputs);
    }

    // The actual params-file-based command line executed for a compile action.
    Artifact javaBuilderJar = checkNotNull(javaBuilder.getExecutable());
    if (!javaBuilderJar.getExtension().equals("jar")) {
      // JavaBuilder is a non-deploy.jar executable.
      builder.setExecutable(javaBuilder);
    } else if (!instrumentationJars.isEmpty()) {
      builder.setExecutable(javaExecutable);
      builder.addTool(javaBuilderJar);
      builder.addExecutableArguments(javacJvmOpts);
      builder.addExecutableArguments(
          "-cp",
          Stream.concat(instrumentationJars.stream(), Stream.of(javaBuilderJar))
              .map(Artifact::getExecPathString)
              .collect(joining(ruleContext.getConfiguration().getHostPathSeparator())));
      builder.addExecutableArguments(javaSemantics.getJavaBuilderMainClass());
    } else {
      // If there are no instrumentation jars, use simpler '-jar' option to launch JavaBuilder.
      builder.setJarExecutable(javaExecutable, javaBuilderJar, javacJvmOpts);
    }

    if (artifactForExperimentalCoverage != null) {
      ruleContext.registerAction(
          new LazyWritePathsFileAction(
              ruleContext.getActionOwner(), artifactForExperimentalCoverage, sourceFiles, false));
    }

    builder.addTool(langtoolsJar);
    builder.addTransitiveTools(toolsJars);
    builder.addTools(instrumentationJars);

    builder.addTransitiveInputs(classpathEntries);
    builder.addTransitiveInputs(compileTimeDependencyArtifacts);
    builder.addTransitiveInputs(plugins.processorClasspath());
    builder.addTransitiveInputs(plugins.data());
    builder.addTransitiveInputs(extraData);
    builder.addInputs(sourceJars);
    builder.addInputs(sourceFiles);
    builder.addInputs(javabaseInputs);
    builder.addInputs(bootclasspathEntries);
    builder.addInputs(sourcePathEntries);
    builder.addInputs(extdirInputs);
    if (artifactForExperimentalCoverage != null) {
      builder.addInput(artifactForExperimentalCoverage);
    }

    CustomCommandLine commandLine =
        buildParamFileContents(ruleContext.getConfiguration(), internedJcopts);
    builder.addCommandLine(
        commandLine,
        ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
            .setCharset(ISO_8859_1)
            .setUseAlways(true)
            .build());

    builder.setProgressMessage(getProgressMessage());
    builder.setMnemonic(MNEMONIC);
    builder.setResources(LOCAL_RESOURCES);
    builder.setEnvironment(UTF8_ENVIRONMENT);
    builder.setExecutionInfo(executionInfo);

    builder.setExtraActionInfo(
        new JavaCompileExtraActionInfoSupplier(
            outputJar,
            classpathEntries,
            bootclasspathEntries,
            plugins.processorClasspath(),
            plugins.processorClasses(),
            sourceJars,
            sourceFiles,
            internedJcopts,
            commandLine));

    ruleContext.getAnalysisEnvironment().registerAction(builder.build(ruleContext));
  }

  private CustomCommandLine buildParamFileContents(
      BuildConfiguration configuration, Collection<String> javacOpts) {
    checkNotNull(classDirectory, "classDirectory should not be null");
    checkNotNull(tempDirectory, "tempDirectory should not be null");

    CustomCommandLine.Builder result = CustomCommandLine.builder();

    result.addPath("--classdir", classDirectory);
    result.addPath("--tempdir", tempDirectory);
    result.addExecPath("--output", outputJar);
    result.addExecPath("--native_header_output", nativeHeaderOutput);
    result.addPath("--sourcegendir", sourceGenDirectory);
    result.addExecPath("--generated_sources_output", gensrcOutputJar);
    result.addExecPath("--output_manifest_proto", manifestProtoOutput);
    if (compressJar) {
      result.add("--compress_jar");
    }
    result.addExecPath("--output_deps_proto", outputDepsProto);
    result.addExecPaths("--extclasspath", extdirInputs);
    result.addExecPaths("--bootclasspath", bootclasspathEntries);
    result.addExecPaths("--sourcepath", sourcePathEntries);
    result.addExecPaths("--processorpath", plugins.processorClasspath());
    result.addAll("--processors", plugins.processorClasses());
    result.addExecPaths("--source_jars", ImmutableList.copyOf(sourceJars));
    result.addExecPaths("--sources", sourceFiles);
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
    result.add("--injecting_rule_kind", injectingRuleKind);
    result.addExecPaths("--classpath", classpathEntries);
    // strict_java_deps controls whether the mapping from jars to targets is
    // written out and whether we try to minimize the compile-time classpath.
    if (strictJavaDeps != StrictDepsMode.OFF) {
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
    if (plugins.processorClasses().isEmpty()) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    List<String> shortNames = new ArrayList<>();
    for (String name : plugins.processorClasses()) {
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

  public JavaCompileActionBuilder setJavaExecutable(PathFragment javaExecutable) {
    this.javaExecutable = javaExecutable;
    return this;
  }

  public JavaCompileActionBuilder setJavaBaseInputs(Iterable<Artifact> javabaseInputs) {
    this.javabaseInputs = ImmutableList.copyOf(javabaseInputs);
    return this;
  }

  public JavaCompileActionBuilder setOutputJar(Artifact outputJar) {
    this.outputJar = outputJar;
    return this;
  }

  public JavaCompileActionBuilder setNativeHeaderOutput(Artifact nativeHeaderOutput) {
    this.nativeHeaderOutput = nativeHeaderOutput;
    return this;
  }

  public JavaCompileActionBuilder setGensrcOutputJar(Artifact gensrcOutputJar) {
    this.gensrcOutputJar = gensrcOutputJar;
    return this;
  }

  public JavaCompileActionBuilder setManifestProtoOutput(Artifact manifestProtoOutput) {
    this.manifestProtoOutput = manifestProtoOutput;
    return this;
  }

  public JavaCompileActionBuilder setOutputDepsProto(Artifact outputDepsProto) {
    this.outputDepsProto = outputDepsProto;
    return this;
  }

  public JavaCompileActionBuilder setAdditionalOutputs(Collection<Artifact> outputs) {
    this.additionalOutputs = outputs;
    return this;
  }

  public JavaCompileActionBuilder setMetadata(Artifact metadata) {
    this.metadata = metadata;
    return this;
  }

  public JavaCompileActionBuilder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
    this.sourceFiles = sourceFiles;
    return this;
  }

  public JavaCompileActionBuilder setSourceJars(ImmutableList<Artifact> sourceJars) {
    checkState(this.sourceJars.isEmpty());
    this.sourceJars = checkNotNull(sourceJars, "sourceJars must not be null");
    return this;
  }

  /**
   * Sets the strictness of Java dependency checking, see {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfiguration.StrictDepsMode}.
   */
  public JavaCompileActionBuilder setStrictJavaDeps(StrictDepsMode strictDeps) {
    strictJavaDeps = strictDeps;
    return this;
  }

  /** Sets the tool with which to fix dependency errors. */
  public JavaCompileActionBuilder setFixDepsTool(String depsTool) {
    fixDepsTool = depsTool;
    return this;
  }

  /** Accumulates the given jar artifacts as being provided by direct dependencies. */
  public JavaCompileActionBuilder setDirectJars(NestedSet<Artifact> directJars) {
    this.directJars = checkNotNull(directJars, "directJars must not be null");
    return this;
  }

  public JavaCompileActionBuilder setCompileTimeDependencyArtifacts(
      NestedSet<Artifact> dependencyArtifacts) {
    checkNotNull(compileTimeDependencyArtifacts, "dependencyArtifacts must not be null");
    this.compileTimeDependencyArtifacts = dependencyArtifacts;
    return this;
  }

  public JavaCompileActionBuilder setJavacOpts(Iterable<String> copts) {
    this.javacOpts = ImmutableList.copyOf(copts);
    return this;
  }

  public JavaCompileActionBuilder setJavacJvmOpts(ImmutableList<String> opts) {
    this.javacJvmOpts = opts;
    return this;
  }

  public JavaCompileActionBuilder setJavacExecutionInfo(
      ImmutableMap<String, String> executionInfo) {
    this.executionInfo = executionInfo;
    return this;
  }

  public JavaCompileActionBuilder setCompressJar(boolean compressJar) {
    this.compressJar = compressJar;
    return this;
  }

  public JavaCompileActionBuilder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
    this.classpathEntries = classpathEntries;
    return this;
  }

  public JavaCompileActionBuilder setBootclasspathEntries(Iterable<Artifact> bootclasspathEntries) {
    this.bootclasspathEntries = ImmutableList.copyOf(bootclasspathEntries);
    return this;
  }

  public JavaCompileActionBuilder setSourcePathEntries(Iterable<Artifact> sourcePathEntries) {
    this.sourcePathEntries = ImmutableList.copyOf(sourcePathEntries);
    return this;
  }

  public JavaCompileActionBuilder setExtdirInputs(Iterable<Artifact> extdirEntries) {
    this.extdirInputs = ImmutableList.copyOf(extdirEntries);
    return this;
  }

  /** Sets the directory where source files generated by annotation processors should be stored. */
  public JavaCompileActionBuilder setSourceGenDirectory(PathFragment sourceGenDirectory) {
    this.sourceGenDirectory = sourceGenDirectory;
    return this;
  }

  public JavaCompileActionBuilder setTempDirectory(PathFragment tempDirectory) {
    this.tempDirectory = tempDirectory;
    return this;
  }

  public JavaCompileActionBuilder setClassDirectory(PathFragment classDirectory) {
    this.classDirectory = classDirectory;
    return this;
  }

  public JavaCompileActionBuilder setPlugins(JavaPluginInfo plugins) {
    checkNotNull(plugins, "plugins must not be null");
    checkState(this.plugins.isEmpty());
    this.plugins = plugins;
    return this;
  }

  public void setExtraData(NestedSet<Artifact> extraData) {
    checkNotNull(extraData, "extraData must not be null");
    checkState(this.extraData.isEmpty());
    this.extraData = extraData;
  }

  public JavaCompileActionBuilder setLangtoolsJar(Artifact langtoolsJar) {
    this.langtoolsJar = langtoolsJar;
    return this;
  }

  /** Sets the tools jars. */
  public JavaCompileActionBuilder setToolsJars(NestedSet<Artifact> toolsJars) {
    checkNotNull(toolsJars, "toolsJars must not be null");
    this.toolsJars = toolsJars;
    return this;
  }

  public JavaCompileActionBuilder setJavaBuilder(FilesToRunProvider javaBuilder) {
    this.javaBuilder = javaBuilder;
    return this;
  }

  public JavaCompileActionBuilder setInstrumentationJars(Iterable<Artifact> instrumentationJars) {
    this.instrumentationJars = ImmutableList.copyOf(instrumentationJars);
    return this;
  }

  public JavaCompileActionBuilder setArtifactForExperimentalCoverage(
      Artifact artifactForExperimentalCoverage) {
    this.artifactForExperimentalCoverage = artifactForExperimentalCoverage;
    return this;
  }

  public JavaCompileActionBuilder setTargetLabel(Label targetLabel) {
    this.targetLabel = targetLabel;
    return this;
  }

  public JavaCompileActionBuilder setInjectingRuleKind(@Nullable String injectingRuleKind) {
    this.injectingRuleKind = injectingRuleKind;
    return this;
  }
}
