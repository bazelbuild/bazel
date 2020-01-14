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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.JavaClasspathMode;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider.JavaPluginInfo;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A helper class for compiling Java targets. It contains method to create the various intermediate
 * Artifacts for using ijars and source ijars.
 *
 * <p>Also supports the creation of resource and source only Jars.
 */
public final class JavaCompilationHelper {

  private final RuleContext ruleContext;
  private final JavaToolchainProvider javaToolchain;
  private final JavaRuntimeInfo hostJavabase;
  private final JavaTargetAttributes.Builder attributes;
  private JavaTargetAttributes builtAttributes;
  private final ImmutableList<String> customJavacOpts;
  private final ImmutableList<String> customJavacJvmOpts;
  private final List<Artifact> translations = new ArrayList<>();
  private boolean translationsFrozen;
  private final JavaSemantics semantics;
  private final ImmutableList<Artifact> additionalJavaBaseInputs;
  private final StrictDepsMode strictJavaDeps;
  private final String fixDepsTool;
  private NestedSet<Artifact> localClassPathEntries = NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  private static final PathFragment JAVAC = PathFragment.create("_javac");

  private JavaCompilationHelper(
      RuleContext ruleContext,
      JavaSemantics semantics,
      ImmutableList<String> javacOpts,
      JavaTargetAttributes.Builder attributes,
      JavaToolchainProvider javaToolchainProvider,
      JavaRuntimeInfo hostJavabase,
      ImmutableList<Artifact> additionalJavaBaseInputs,
      boolean disableStrictDeps) {
    this.ruleContext = ruleContext;
    this.javaToolchain = Preconditions.checkNotNull(javaToolchainProvider);
    this.hostJavabase = Preconditions.checkNotNull(hostJavabase);
    this.attributes = attributes;
    this.customJavacOpts = javacOpts;
    this.customJavacJvmOpts = javaToolchain.getJavabuilderJvmOptions();
    this.semantics = semantics;
    this.additionalJavaBaseInputs = additionalJavaBaseInputs;
    this.strictJavaDeps =
        disableStrictDeps ? StrictDepsMode.OFF : getJavaConfiguration().getFilteredStrictJavaDeps();
    this.fixDepsTool = getJavaConfiguration().getFixDepsTool();
  }

  public JavaCompilationHelper(
      RuleContext ruleContext,
      JavaSemantics semantics,
      ImmutableList<String> javacOpts,
      JavaTargetAttributes.Builder attributes,
      JavaToolchainProvider javaToolchainProvider,
      JavaRuntimeInfo hostJavabase,
      ImmutableList<Artifact> additionalJavaBaseInputs) {
    this(
        ruleContext,
        semantics,
        javacOpts,
        attributes,
        javaToolchainProvider,
        hostJavabase,
        additionalJavaBaseInputs,
        false);
  }

  public JavaCompilationHelper(
      RuleContext ruleContext,
      JavaSemantics semantics,
      ImmutableList<String> javacOpts,
      JavaTargetAttributes.Builder attributes) {
    this(
        ruleContext,
        semantics,
        javacOpts,
        attributes,
        JavaToolchainProvider.from(ruleContext),
        JavaRuntimeInfo.forHost(ruleContext),
        ImmutableList.of());
  }

  public JavaCompilationHelper(
      RuleContext ruleContext,
      JavaSemantics semantics,
      ImmutableList<String> javacOpts,
      JavaTargetAttributes.Builder attributes,
      ImmutableList<Artifact> additionalJavaBaseInputs,
      boolean disableStrictDeps) {
    this(
        ruleContext,
        semantics,
        javacOpts,
        attributes,
        JavaToolchainProvider.from(ruleContext),
        JavaRuntimeInfo.forHost(ruleContext),
        additionalJavaBaseInputs,
        disableStrictDeps);
  }

  @VisibleForTesting
  JavaCompilationHelper(
      RuleContext ruleContext, JavaSemantics semantics, JavaTargetAttributes.Builder attributes) {
    this(ruleContext, semantics, getDefaultJavacOptsFromRule(ruleContext), attributes);
  }

  JavaTargetAttributes getAttributes() {
    if (builtAttributes == null) {
      builtAttributes = attributes.build();
      if (!localClassPathEntries.isEmpty()) {
        builtAttributes = builtAttributes.withAdditionalClassPathEntries(localClassPathEntries);
      }
    }
    return builtAttributes;
  }

  public RuleContext getRuleContext() {
    return ruleContext;
  }

  private AnalysisEnvironment getAnalysisEnvironment() {
    return ruleContext.getAnalysisEnvironment();
  }

  private BuildConfiguration getConfiguration() {
    return ruleContext.getConfiguration();
  }

  private JavaConfiguration getJavaConfiguration() {
    return ruleContext.getFragment(JavaConfiguration.class);
  }

  public JavaCompileOutputs<Artifact> createOutputs(Artifact output) {
    JavaCompileOutputs.Builder<Artifact> builder =
        JavaCompileOutputs.builder()
            .output(output)
            .manifestProto(
                deriveOutput(
                    output,
                    FileSystemUtils.appendExtension(
                        output.getRootRelativePath(), "_manifest_proto")))
            .nativeHeader(deriveOutput(output, "-native-header"));
    if (generatesOutputDeps()) {
      builder.depsProto(
          deriveOutput(
              output, FileSystemUtils.replaceExtension(output.getRootRelativePath(), ".jdeps")));
    }
    if (usesAnnotationProcessing()) {
      builder.genClass(deriveOutput(output, "-gen")).genSource(deriveOutput(output, "-gensrc"));
    }
    JavaCompileOutputs<Artifact> result = builder.build();
    return result;
  }

  public void createCompileAction(JavaCompileOutputs<Artifact> outputs)
      throws InterruptedException {
    if (outputs.genClass() != null) {
      createGenJarAction(
          outputs.output(), outputs.manifestProto(), outputs.genClass(), hostJavabase);
    }

    JavaTargetAttributes attributes = getAttributes();

    if (separateResourceJar(attributes)) {
      Artifact originalOutput = outputs.output();
      outputs =
          outputs.withOutput(
              ruleContext.getDerivedArtifact(
                  FileSystemUtils.appendWithoutExtension(
                      outputs.output().getRootRelativePath(), "-class"),
                  outputs.output().getRoot()));
      createResourceJarAction(originalOutput, ImmutableList.of(outputs.output()));
    }

    JavaCompileActionBuilder builder =
        new JavaCompileActionBuilder(
            ruleContext.getActionOwner(), ruleContext.getConfiguration(), javaToolchain);
    builder.setJavaExecutable(hostJavabase.javaBinaryExecPathFragment());
    builder.setJavaBaseInputs(
        NestedSetBuilder.fromNestedSet(hostJavabase.javaBaseInputsMiddleman())
            .addAll(additionalJavaBaseInputs)
            .build());
    Label label = ruleContext.getLabel();
    builder.setTargetLabel(label);
    Artifact coverageArtifact = maybeCreateCoverageArtifact(outputs.output());
    builder.setCoverageArtifact(coverageArtifact);
    builder.setClasspathEntries(attributes.getCompileTimeClassPath());
    builder.setBootclasspathEntries(getBootclasspathOrDefault());
    builder.setSourcePathEntries(attributes.getSourcePath());
    builder.setExtdirInputs(getExtdirInputs());
    builder.setToolsJars(javaToolchain.getTools());
    builder.setJavaBuilder(javaToolchain.getJavaBuilder());
    builder.setGenSourceOutput(outputs.genSource());
    builder.setOutputs(outputs);

    builder.setAdditionalOutputs(attributes.getAdditionalOutputs());
    ImmutableSet<Artifact> sourceFiles = attributes.getSourceFiles();
    builder.setSourceFiles(sourceFiles);
    builder.setSourceJars(attributes.getSourceJars());
    builder.setJavacOpts(customJavacOpts);
    builder.setJavacJvmOpts(customJavacJvmOpts);
    builder.setJavacExecutionInfo(getExecutionInfo());
    builder.setCompressJar(true);
    builder.setSourceGenDirectory(sourceGenDir(outputs.output(), label));
    builder.setTempDirectory(tempDir(outputs.output(), label));
    builder.setClassDirectory(classDir(outputs.output(), label));
    builder.setPlugins(attributes.plugins().plugins());
    builder.setBuiltinProcessorNames(javaToolchain.getHeaderCompilerBuiltinProcessors());
    builder.setExtraData(JavaCommon.computePerPackageData(ruleContext, javaToolchain));
    builder.setStrictJavaDeps(attributes.getStrictJavaDeps());
    builder.setFixDepsTool(getJavaConfiguration().getFixDepsTool());
    builder.setDirectJars(attributes.getDirectJars());
    builder.setCompileTimeDependencyArtifacts(attributes.getCompileTimeDependencyArtifacts());
    builder.setTargetLabel(
        attributes.getTargetLabel() == null ? label : attributes.getTargetLabel());
    builder.setInjectingRuleKind(attributes.getInjectingRuleKind());

    if (coverageArtifact != null) {
      ruleContext.registerAction(
          new LazyWritePathsFileAction(
              ruleContext.getActionOwner(), coverageArtifact, sourceFiles, false));
    }

    JavaCompileAction javaCompileAction = builder.build();
    ruleContext.getAnalysisEnvironment().registerAction(javaCompileAction);
  }

  /**
   * If there are sources and no resource, the only output is from the javac action. Otherwise
   * create a separate jar for the compilation and add resources with singlejar.
   */
  private boolean separateResourceJar(JavaTargetAttributes attributes) {
    return !attributes.getResources().isEmpty()
        || !attributes.getResourceJars().isEmpty()
        || !attributes.getClassPathResources().isEmpty()
        || !getTranslations().isEmpty();
  }

  private ImmutableMap<String, String> getExecutionInfo() throws InterruptedException {
    ImmutableMap.Builder<String, String> executionInfo = ImmutableMap.builder();
    executionInfo.putAll(
        getConfiguration()
            .modifiedExecutionInfo(
                javaToolchain.getJavacSupportsWorkers()
                    ? ExecutionRequirements.WORKER_MODE_ENABLED
                    : ImmutableMap.of(),
                JavaCompileActionBuilder.MNEMONIC));
    executionInfo.putAll(
        TargetUtils.getExecutionInfo(ruleContext.getRule(), ruleContext.isAllowTagsPropagation()));

    return executionInfo.build();
  }

  /** Returns the bootclasspath explicit set in attributes if present, or else the default. */
  public NestedSet<Artifact> getBootclasspathOrDefault() {
    JavaTargetAttributes attributes = getAttributes();
    if (!attributes.getBootClassPath().isEmpty()) {
      return attributes.getBootClassPath();
    } else {
      return getBootClasspath(javaToolchain);
    }
  }

  public boolean addCoverageSupport() {
    FilesToRunProvider jacocoRunner = javaToolchain.getJacocoRunner();
    if (jacocoRunner == null) {
      return false;
    }
    Artifact jacocoRunnerJar = jacocoRunner.getExecutable();
    if (isStrict()) {
      attributes.addDirectJar(jacocoRunnerJar);
    }
    attributes.addCompileTimeClassPathEntry(jacocoRunnerJar);
    attributes.addRuntimeClassPathEntry(jacocoRunnerJar);
    return true;
  }

  /**
   * Creates an {@link Artifact} needed by {@code JacocoCoverageRunner}.
   *
   * <p>The {@link Artifact} is created in the same directory as the given {@code compileJar} and
   * has the suffix {@code -paths-for-coverage.txt}.
   *
   * <p>Returns {@code null} if {@code compileJar} should not be instrumented.
   */
  private Artifact maybeCreateCoverageArtifact(Artifact compileJar) {
    if (!shouldInstrumentJar()) {
      return null;
    }
    PathFragment packageRelativePath =
        compileJar.getRootRelativePath().relativeTo(ruleContext.getPackageDirectory());
    PathFragment path =
        FileSystemUtils.replaceExtension(packageRelativePath, "-paths-for-coverage.txt");
    return ruleContext.getPackageRelativeArtifact(path, compileJar.getRoot());
  }

  private boolean shouldInstrumentJar() {
    RuleContext ruleContext = getRuleContext();
    return getConfiguration().isCodeCoverageEnabled()
        && attributes.hasSources()
        && InstrumentedFilesCollector.shouldIncludeLocalSources(
            ruleContext.getConfiguration(), ruleContext.getLabel(), ruleContext.isTestTarget());
  }

  private boolean shouldUseHeaderCompilation() {
    if (!getJavaConfiguration().useHeaderCompilation()) {
      return false;
    }
    if (!attributes.hasSources()) {
      return false;
    }
    if (javaToolchain.getForciblyDisableHeaderCompilation()) {
      return false;
    }
    if (javaToolchain.getHeaderCompiler() == null) {
      getRuleContext()
          .ruleError(
              String.format(
                  "header compilation was requested but it is not supported by the current Java"
                      + " toolchain '%s'; see the java_toolchain.header_compiler attribute",
                  javaToolchain.getToolchainLabel()));
      return false;
    }
    if (getJavaConfiguration().requireJavaToolchainHeaderCompilerDirect()
        && javaToolchain.getHeaderCompilerDirect() == null) {
      getRuleContext()
          .ruleError(
              String.format(
                  "header compilation was requested but it is not supported by the current Java"
                      + " toolchain '%s'; see the java_toolchain.header_compiler_direct attribute",
                  javaToolchain.getToolchainLabel()));
      return false;
    }
    return true;
  }

  private Artifact turbineOutput(Artifact classJar, String newExtension) {
    return getAnalysisEnvironment()
        .getDerivedArtifact(
            FileSystemUtils.replaceExtension(classJar.getRootRelativePath(), newExtension),
            classJar.getRoot());
  }

  /**
   * Creates the Action that compiles ijars from source.
   *
   * @param runtimeJar the jar output of this java compilation, used to create output-relative paths
   *     for new artifacts.
   */
  private Artifact createHeaderCompilationAction(
      Artifact runtimeJar, JavaCompilationArtifacts.Builder artifactBuilder)
      throws InterruptedException {

    Artifact headerJar = turbineOutput(runtimeJar, "-hjar.jar");
    Artifact headerDeps = turbineOutput(runtimeJar, "-hjar.jdeps");

    JavaTargetAttributes attributes = getAttributes();
    JavaHeaderCompileActionBuilder builder = new JavaHeaderCompileActionBuilder(getRuleContext());
    builder.setSourceFiles(attributes.getSourceFiles());
    builder.setSourceJars(attributes.getSourceJars());
    builder.setClasspathEntries(attributes.getCompileTimeClassPath());
    builder.setBootclasspathEntries(
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(getBootclasspathOrDefault())
            .addTransitive(getExtdirInputs())
            .build());

    // only run API-generating annotation processors during header compilation
    JavaPluginInfo plugins = attributes.plugins().apiGeneratingPlugins();
    builder.setPlugins(plugins);
    // Exclude any per-package configured data (see JavaCommon.computePerPackageData).
    // It is used to allow Error Prone checks to load additional data,
    // and Error Prone doesn't run during header compilation.
    builder.addAllJavacOpts(getJavacOpts());
    if (plugins
        .processorClasses()
        .toList()
        .contains("dagger.internal.codegen.ComponentProcessor")) {
      // see b/31371210
      builder.addJavacOpt("-Aexperimental_turbine_hjar");
    }
    builder.setOutputJar(headerJar);
    builder.setOutputDepsProto(headerDeps);
    builder.setStrictJavaDeps(attributes.getStrictJavaDeps());
    builder.setCompileTimeDependencyArtifacts(attributes.getCompileTimeDependencyArtifacts());
    builder.setDirectJars(attributes.getDirectJars());
    builder.setTargetLabel(attributes.getTargetLabel());
    builder.setInjectingRuleKind(attributes.getInjectingRuleKind());
    builder.setAdditionalInputs(NestedSetBuilder.wrap(Order.LINK_ORDER, additionalJavaBaseInputs));
    builder.setToolsJars(javaToolchain.getTools());
    builder.build(javaToolchain, hostJavabase);

    artifactBuilder.setCompileTimeDependencies(headerDeps);
    return headerJar;
  }

  private Artifact deriveOutput(Artifact outputJar, String suffix) {
    return deriveOutput(
        outputJar, FileSystemUtils.appendWithoutExtension(outputJar.getRootRelativePath(), suffix));
  }

  private Artifact deriveOutput(Artifact outputJar, PathFragment path) {
    return getRuleContext().getDerivedArtifact(path, outputJar.getRoot());
  }

  /** Returns whether this target uses annotation processing. */
  public boolean usesAnnotationProcessing() {
    JavaTargetAttributes attributes = getAttributes();
    return getJavacOpts().contains("-processor") || attributes.plugins().hasProcessors();
  }

  public void createGenJarAction(
      Artifact classJar,
      Artifact manifestProto,
      Artifact genClassJar,
      JavaRuntimeInfo hostJavabase) {
    getRuleContext()
        .registerAction(
            new SpawnAction.Builder()
                .addInput(manifestProto)
                .addInput(classJar)
                .addOutput(genClassJar)
                .addTransitiveInputs(hostJavabase.javaBaseInputsMiddleman())
                .setJarExecutable(
                    JavaCommon.getHostJavaExecutable(hostJavabase),
                    getGenClassJar(ruleContext),
                    javaToolchain.getJvmOptions())
                .addCommandLine(
                    CustomCommandLine.builder()
                        .addExecPath("--manifest_proto", manifestProto)
                        .addExecPath("--class_jar", classJar)
                        .addExecPath("--output_jar", genClassJar)
                        .add("--temp_dir")
                        .addPath(tempDir(genClassJar, ruleContext.getLabel()))
                        .build())
                .setProgressMessage("Building genclass jar %s", genClassJar.prettyPrint())
                .setMnemonic("JavaSourceJar")
                .build(getRuleContext()));
  }

  /** Returns the GenClass deploy jar Artifact. */
  private Artifact getGenClassJar(RuleContext ruleContext) {
    Artifact genClass = javaToolchain.getGenClass();
    if (genClass != null) {
      return genClass;
    }
    return ruleContext.getPrerequisiteArtifact("$genclass", Mode.HOST);
  }

  /**
   * Returns whether this target emits dependency information. Compilation must occur, so certain
   * targets acting as aliases have to be filtered out.
   */
  private boolean generatesOutputDeps() {
    return getJavaConfiguration().getGenerateJavaDeps() && attributes.hasSources();
  }

  /**
   * Creates and registers an Action that packages all of the resources into a Jar. This includes
   * the declared resources, the classpath resources and the translated messages.
   */
  public void createResourceJarAction(Artifact resourceJar) {
    createResourceJarAction(resourceJar, ImmutableList.<Artifact>of());
  }

  private void createResourceJarAction(Artifact resourceJar, ImmutableList<Artifact> extraJars) {
    checkNotNull(resourceJar, "resource jar output must not be null");
    JavaTargetAttributes attributes = getAttributes();
    new ResourceJarActionBuilder()
        .setHostJavaRuntime(hostJavabase)
        .setAdditionalInputs(NestedSetBuilder.wrap(Order.STABLE_ORDER, additionalJavaBaseInputs))
        .setJavaToolchain(javaToolchain)
        .setOutputJar(resourceJar)
        .setResources(attributes.getResources())
        .setClasspathResources(attributes.getClassPathResources())
        .setTranslations(getTranslations())
        .setResourceJars(
            NestedSetBuilder.fromNestedSet(attributes.getResourceJars()).addAll(extraJars).build())
        .build(semantics, ruleContext);
  }

  /**
   * Produces a derived directory where source files generated by annotation processors should be
   * stored.
   */
  private static PathFragment sourceGenDir(Artifact outputJar, Label label) {
    return workDir(outputJar, "_sourcegenfiles", label);
  }

  private static PathFragment tempDir(Artifact outputJar, Label label) {
    return workDir(outputJar, "_temp", label);
  }

  /** Produces a derived directory where class outputs should be stored. */
  public static PathFragment classDir(Artifact outputJar, Label label) {
    return workDir(outputJar, "_classes", label);
  }

  /**
   * For an output jar and a suffix, produces a derived directory under the same root as {@code
   * label} with directory name composed of {@code outputJar}'s basename and {@code suffix}.
   *
   * <p>Note that this won't work if a rule produces two jars with the same basename.
   */
  private static PathFragment workDir(Artifact outputJar, String suffix, Label label) {
    String basename = FileSystemUtils.removeExtension(outputJar.getExecPath().getBaseName());
    return outputJar
        .getRoot()
        .getExecPath()
        .getRelative(AnalysisUtils.getUniqueDirectory(label, JAVAC))
        .getRelative(basename + suffix);
  }

  /**
   * Creates an Action that packages the Java source files into a Jar. If {@code gensrcJar} is
   * non-null, includes the contents of the {@code gensrcJar} with the output source jar.
   *
   * @param outputJar the Artifact to create with the Action
   * @param gensrcJar the generated sources jar Artifact that should be included with the sources in
   *     the output Artifact. May be null.
   * @param javaToolchainProvider is used by SingleJarActionBuilder to retrieve jvm options
   * @param hostJavabase the Java runtime used to run the binaries in the action
   */
  public void createSourceJarAction(
      Artifact outputJar,
      @Nullable Artifact gensrcJar,
      JavaToolchainProvider javaToolchainProvider,
      JavaRuntimeInfo hostJavabase) {
    JavaTargetAttributes attributes = getAttributes();
    NestedSetBuilder<Artifact> resourceJars = NestedSetBuilder.stableOrder();
    resourceJars.addAll(attributes.getSourceJars());
    if (gensrcJar != null) {
      resourceJars.add(gensrcJar);
    }
    SingleJarActionBuilder.createSourceJarAction(
        ruleContext,
        ruleContext,
        semantics,
        NestedSetBuilder.<Artifact>wrap(Order.STABLE_ORDER, attributes.getSourceFiles()),
        resourceJars.build(),
        outputJar,
        javaToolchainProvider,
        hostJavabase);
  }

  public void createSourceJarAction(Artifact outputJar, @Nullable Artifact gensrcJar) {
    JavaTargetAttributes attributes = getAttributes();
    NestedSetBuilder<Artifact> resourceJars = NestedSetBuilder.stableOrder();
    resourceJars.addAll(attributes.getSourceJars());
    if (gensrcJar != null) {
      resourceJars.add(gensrcJar);
    }
    SingleJarActionBuilder.createSourceJarAction(
        ruleContext,
        semantics,
        NestedSetBuilder.<Artifact>wrap(Order.STABLE_ORDER, attributes.getSourceFiles()),
        resourceJars.build(),
        outputJar);
  }

  /**
   * Creates the actions that produce the interface jar. Adds the jar artifacts to the given
   * JavaCompilationArtifacts builder.
   *
   * @return the header jar (if requested), or ijar (if requested), or else the class jar
   */
  public Artifact createCompileTimeJarAction(
      Artifact runtimeJar, JavaCompilationArtifacts.Builder builder) throws InterruptedException {
    Artifact jar;
    boolean isFullJar = false;
    if (shouldUseHeaderCompilation()) {
      jar = createHeaderCompilationAction(runtimeJar, builder);
    } else if (getJavaConfiguration().getUseIjars()) {
      JavaTargetAttributes attributes = getAttributes();
      jar =
          createIjarAction(
              ruleContext,
              javaToolchain,
              runtimeJar,
              attributes.getTargetLabel(),
              attributes.getInjectingRuleKind(),
              false);
    } else {
      jar = runtimeJar;
      isFullJar = true;
    }
    if (isFullJar) {
      builder.addCompileTimeJarAsFullJar(jar);
    } else {
      builder.addInterfaceJarWithFullJar(jar, runtimeJar);
    }
    return jar;
  }

  private void addArgsAndJarsToAttributes(
      JavaCompilationArgsProvider args, NestedSet<Artifact> directJars) {
    // Can only be non-null when isStrict() returns true.
    if (directJars != null) {
      attributes.addDirectJars(directJars);
    }

    attributes.merge(args);
  }

  /**
   * Adds compile-time dependencies that will be included on the classpath, but which will not be
   * visible to targets that depend on the current compilation.
   */
  public void addLocalClassPathEntries(NestedSet<Artifact> localClassPathEntries) {
    checkState(
        builtAttributes == null,
        "addLocalClassPathEntries must be called before the first call to getAttributes()");
    this.localClassPathEntries = localClassPathEntries;
  }

  private void addLibrariesToAttributesInternal(Iterable<? extends TransitiveInfoCollection> deps) {
    JavaCompilationArgsProvider args = JavaCompilationArgsProvider.legacyFromTargets(deps);

    NestedSet<Artifact> directJars =
        isStrict() ? getNonRecursiveCompileTimeJarsFromCollection(deps) : null;
    addArgsAndJarsToAttributes(args, directJars);
  }

  private boolean isStrict() {
    return getStrictJavaDeps() != StrictDepsMode.OFF;
  }

  private NestedSet<Artifact> getNonRecursiveCompileTimeJarsFromCollection(
      Iterable<? extends TransitiveInfoCollection> deps) {
    return JavaCompilationArgsProvider.legacyFromTargets(deps).getDirectCompileTimeJars();
  }

  static void addDependencyArtifactsToAttributes(
      JavaTargetAttributes.Builder attributes,
      Iterable<? extends JavaCompilationArgsProvider> deps) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    for (JavaCompilationArgsProvider provider : deps) {
      result.addTransitive(provider.getCompileTimeJavaDependencyArtifacts());
    }
    attributes.addCompileTimeDependencyArtifacts(result.build());
  }

  /**
   * Adds the compile time and runtime Java libraries in the transitive closure of the deps to the
   * attributes.
   *
   * @param deps the dependencies to be included as roots of the transitive closure
   */
  public void addLibrariesToAttributes(Iterable<? extends TransitiveInfoCollection> deps) {
    // Enforcing strict Java dependencies: when the --strict_java_deps flag is
    // WARN or ERROR, or is DEFAULT and strict_java_deps attribute is unset,
    // we use a stricter javac compiler to perform direct deps checks.
    attributes.setStrictJavaDeps(getStrictJavaDeps());
    addLibrariesToAttributesInternal(deps);

    JavaClasspathMode classpathMode = getJavaConfiguration().getReduceJavaClasspath();
    if (isStrict() && classpathMode != JavaClasspathMode.OFF) {
      List<JavaCompilationArgsProvider> compilationArgsProviders = new ArrayList<>();
      for (TransitiveInfoCollection dep : deps) {
        JavaCompilationArgsProvider provider =
            JavaInfo.getProvider(JavaCompilationArgsProvider.class, dep);
        if (provider != null) {
          compilationArgsProviders.add(provider);
        }
      }
      addDependencyArtifactsToAttributes(attributes, compilationArgsProviders);
    }
  }

  /**
   * Determines whether to enable strict_java_deps.
   *
   * @return filtered command line flag value, defaulting to ERROR
   */
  public StrictDepsMode getStrictJavaDeps() {
    return strictJavaDeps;
  }

  /** Determines which tool to use when fixing dependency errors. */
  public String getFixDepsTool() {
    return fixDepsTool;
  }

  /**
   * Gets the value of the "javacopts" attribute combining them with the default options. If the
   * current rule has no javacopts attribute, this method only returns the default options.
   */
  @VisibleForTesting
  public ImmutableList<String> getJavacOpts() {
    return customJavacOpts;
  }

  /**
   * Obtains the standard list of javac opts needed to build {@code rule}.
   *
   * <p>This method must only be called during initialization.
   *
   * @param ruleContext a rule context
   * @return a list of options to provide to javac
   */
  private static ImmutableList<String> getDefaultJavacOptsFromRule(RuleContext ruleContext) {
    return ImmutableList.copyOf(
        Iterables.concat(
            JavaToolchainProvider.from(ruleContext).getJavacOptions(ruleContext),
            ruleContext.getExpander().withDataLocations().tokenized("javacopts")));
  }

  public void setTranslations(Collection<Artifact> translations) {
    Preconditions.checkArgument(!translationsFrozen);
    this.translations.addAll(translations);
  }

  private ImmutableList<Artifact> getTranslations() {
    translationsFrozen = true;
    return ImmutableList.copyOf(translations);
  }

  /**
   * Returns the javac bootclasspath artifacts from the given toolchain (if it has any) or the rule.
   */
  public static NestedSet<Artifact> getBootClasspath(JavaToolchainProvider javaToolchain) {
    return javaToolchain.getBootclasspath();
  }

  /** Returns the extdir artifacts. */
  private final NestedSet<Artifact> getExtdirInputs() {
    return javaToolchain.getExtclasspath();
  }

  /**
   * Creates the Action that creates ijars from Jar files.
   *
   * @param inputJar the Jar to create the ijar for
   * @param addPrefix whether to prefix the path of the generated ijar with the package and name of
   *     the current rule
   * @return the Artifact to create with the Action
   */
  protected static Artifact createIjarAction(
      RuleContext ruleContext,
      JavaToolchainProvider javaToolchain,
      Artifact inputJar,
      @Nullable Label targetLabel,
      @Nullable String injectingRuleKind,
      boolean addPrefix) {
    Artifact interfaceJar = getIjarArtifact(ruleContext, inputJar, addPrefix);
    FilesToRunProvider ijarTarget = javaToolchain.getIjar();
    if (!ruleContext.hasErrors()) {
      CustomCommandLine.Builder commandLine =
          CustomCommandLine.builder().addExecPath(inputJar).addExecPath(interfaceJar);
      if (targetLabel != null) {
        commandLine.addLabel("--target_label", targetLabel);
      }
      if (injectingRuleKind != null) {
        commandLine.add("--injecting_rule_kind", injectingRuleKind);
      }
      ruleContext.registerAction(
          new SpawnAction.Builder()
              .addInput(inputJar)
              .addOutput(interfaceJar)
              .setExecutable(ijarTarget)
              // On Windows, ijar.exe needs msys-2.0.dll and zlib1.dll in PATH.
              // Use default shell environment so that those can be found.
              // TODO(dslomov): revisit this. If ijar is not msys-dependent, this is not needed.
              .useDefaultShellEnvironment()
              .setProgressMessage("Extracting interface %s", ruleContext.getLabel())
              .setMnemonic("JavaIjar")
              .addCommandLine(commandLine.build())
              .build(ruleContext));
    }
    return interfaceJar;
  }

  private static Artifact getIjarArtifact(
      RuleContext ruleContext, Artifact jar, boolean addPrefix) {
    if (addPrefix) {
      PathFragment ruleBase = ruleContext.getUniqueDirectory("_ijar");
      PathFragment artifactDirFragment = jar.getRootRelativePath().getParentDirectory();
      String ijarBasename = FileSystemUtils.removeExtension(jar.getFilename()) + "-ijar.jar";
      return ruleContext.getDerivedArtifact(
          ruleBase.getRelative(artifactDirFragment).getRelative(ijarBasename),
          ruleContext
              .getConfiguration()
              .getGenfilesDirectory(ruleContext.getRule().getRepository()));
    } else {
      return derivedArtifact(ruleContext, jar, "", "-ijar.jar");
    }
  }

  /**
   * Creates a derived artifact from the given artifact by adding the given prefix and removing the
   * extension and replacing it by the given suffix. The new artifact will have the same root as the
   * given one.
   */
  static Artifact derivedArtifact(
      ActionConstructionContext context, Artifact artifact, String prefix, String suffix) {
    PathFragment path = artifact.getRootRelativePath();
    String basename = FileSystemUtils.removeExtension(path.getBaseName()) + suffix;
    path = path.replaceName(prefix + basename);
    return context.getDerivedArtifact(path, artifact.getRoot());
  }
}
