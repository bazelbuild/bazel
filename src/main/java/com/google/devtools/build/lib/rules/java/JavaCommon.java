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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** A helper class to create configured targets for Java rules. */
public class JavaCommon {
  public static final InstrumentationSpec JAVA_COLLECTION_SPEC =
      new InstrumentationSpec(FileTypeSet.of(JavaSemantics.JAVA_SOURCE))
          .withSourceAttributes("srcs")
          .withDependencyAttributes(
              "deps", "data", "resources", "resource_jars", "exports", "runtime_deps", "jars");

  private ClasspathConfiguredFragment classpathFragment = new ClasspathConfiguredFragment();
  private JavaCompilationArtifacts javaArtifacts = JavaCompilationArtifacts.EMPTY;
  private ImmutableList<String> javacOpts;

  // Targets treated as deps in compilation time, runtime time and both
  private final ImmutableMap<ClasspathType, ImmutableList<TransitiveInfoCollection>>
      targetsTreatedAsDeps;

  private final ImmutableList<Artifact> sources;
  private JavaPluginInfo activePlugins = JavaPluginInfo.empty();

  private final RuleContext ruleContext;
  private final JavaSemantics semantics;
  private final JavaToolchainProvider javaToolchain;
  private JavaCompilationHelper javaCompilationHelper;

  public JavaCommon(
      RuleContext ruleContext,
      JavaSemantics semantics,
      ImmutableList<TransitiveInfoCollection> compileDeps,
      ImmutableList<TransitiveInfoCollection> runtimeDeps,
      ImmutableList<TransitiveInfoCollection> bothDeps) {
    this(
        ruleContext,
        semantics,
        ruleContext.getPrerequisiteArtifacts("srcs").list(),
        compileDeps,
        runtimeDeps,
        bothDeps);
  }

  public JavaCommon(
      RuleContext ruleContext,
      JavaSemantics semantics,
      ImmutableList<Artifact> sources,
      ImmutableList<TransitiveInfoCollection> compileDeps,
      ImmutableList<TransitiveInfoCollection> runtimeDeps,
      ImmutableList<TransitiveInfoCollection> bothDeps) {
    this.ruleContext = ruleContext;
    this.javaToolchain = JavaToolchainProvider.from(ruleContext);
    this.semantics = semantics;
    this.sources = sources;
    this.targetsTreatedAsDeps =
        ImmutableMap.of(
            ClasspathType.COMPILE_ONLY, compileDeps,
            ClasspathType.RUNTIME_ONLY, runtimeDeps,
            ClasspathType.BOTH, bothDeps);
  }

  public JavaSemantics getJavaSemantics() {
    return semantics;
  }

  public static ImmutableList<String> getConstraints(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("constraints", Type.STRING_LIST)
        ? ImmutableList.copyOf(ruleContext.attributes().get("constraints", Type.STRING_LIST))
        : ImmutableList.of();
  }

  public void setClassPathFragment(ClasspathConfiguredFragment classpathFragment) {
    this.classpathFragment = classpathFragment;
  }

  public void setJavaCompilationArtifacts(JavaCompilationArtifacts javaArtifacts) {
    this.javaArtifacts = javaArtifacts;
  }

  public JavaCompilationArtifacts getJavaCompilationArtifacts() {
    return javaArtifacts;
  }

  /**
   * Collects Java compilation arguments for this target.
   *
   * @param isNeverLink Whether the target has the 'neverlink' attr.
   * @param srcLessDepsExport If srcs is omitted, deps are exported (deprecated behaviour for
   *     android_library only)
   */
  public JavaCompilationArgsProvider collectJavaCompilationArgs(boolean isNeverLink)
      throws RuleErrorException {
    return collectJavaCompilationArgs(
        /* isNeverLink= */ isNeverLink,
        getJavaCompilationArtifacts(),
        /* deps= */ ImmutableList.of(
            JavaCompilationArgsProvider.legacyFromTargets(
                targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY))),
        /* runtimeDeps= */ ImmutableList.of(
            JavaCompilationArgsProvider.legacyFromTargets(getRuntimeDeps(ruleContext))),
        /* exports= */ ImmutableList.of(
            JavaCompilationArgsProvider.legacyFromTargets(getExports(ruleContext))));
  }

  static JavaCompilationArgsProvider collectJavaCompilationArgs(
      boolean isNeverLink,
      JavaCompilationArtifacts compilationArtifacts,
      List<JavaCompilationArgsProvider> deps,
      List<JavaCompilationArgsProvider> runtimeDeps,
      List<JavaCompilationArgsProvider> exports) {
    ClasspathType type = isNeverLink ? ClasspathType.COMPILE_ONLY : ClasspathType.BOTH;
    JavaCompilationArgsProvider.Builder builder =
        JavaCompilationArgsProvider.builder().merge(compilationArtifacts, isNeverLink);
    exports.forEach(export -> builder.addExports(export, type));
    deps.forEach(dep -> builder.addDeps(dep, type));
    runtimeDeps.forEach(dep -> builder.addDeps(dep, ClasspathType.RUNTIME_ONLY));
    builder.addCompileTimeJavaDependencyArtifacts(
        collectCompileTimeDependencyArtifacts(
            compilationArtifacts.getCompileTimeDependencyArtifact(), exports));
    return builder.build();
  }

  /**
   * Collects Java dependency artifacts for a target.
   *
   * @param jdeps dependency artifact of this target
   * @param exports dependencies with export-like semantics
   */
  public static NestedSet<Artifact> collectCompileTimeDependencyArtifacts(
      @Nullable Artifact jdeps, Collection<JavaCompilationArgsProvider> exports) {
    NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
    if (jdeps != null) {
      builder.add(jdeps);
    }
    exports.stream()
        .map(JavaCompilationArgsProvider::getCompileTimeJavaDependencyArtifacts)
        .forEach(builder::addTransitive);
    return builder.build();
  }

  public static List<TransitiveInfoCollection> getExports(RuleContext ruleContext) {
    // We need to check here because there are classes inheriting from this class that implement
    // rules that don't have this attribute.
    if (ruleContext.attributes().has("exports", BuildType.LABEL_LIST)) {
      return ImmutableList.copyOf(ruleContext.getPrerequisites("exports"));
    } else {
      return ImmutableList.of();
    }
  }

  /**
   * Checks the given runtime dependencies, and emits errors if there is a problem. Also called by
   * {@link #initCommon()} for the current target's runtime dependencies.
   */
  public static void checkRuntimeDeps(
      RuleContext ruleContext, List<TransitiveInfoCollection> runtimeDepInfo)
      throws RuleErrorException {
    for (TransitiveInfoCollection c : runtimeDepInfo) {
      JavaInfo javaInfo = c.get(JavaInfo.PROVIDER);
      if (javaInfo == null) {
        continue;
      }
      boolean reportError =
          !ruleContext.getFragment(JavaConfiguration.class).getAllowRuntimeDepsOnNeverLink();
      if (javaInfo.isNeverlink()) {
        String msg = String.format("neverlink dep %s not allowed in runtime deps", c.getLabel());
        if (reportError) {
          ruleContext.attributeError("runtime_deps", msg);
        } else {
          ruleContext.attributeWarning("runtime_deps", msg);
        }
      }
    }
  }

  /**
   * Collects transitive source jars for the current rule.
   *
   * @param targetSrcJars The source jar artifacts corresponding to the output of the current rule.
   * @return A nested set containing all of the source jar artifacts on which the current rule
   *     transitively depends.
   */
  public NestedSet<Artifact> collectTransitiveSourceJars(Artifact... targetSrcJars)
      throws RuleErrorException {
    return collectTransitiveSourceJars(ImmutableList.copyOf(targetSrcJars));
  }

  /**
   * Collects transitive source jars for the current rule.
   *
   * @param targetSrcJars The source jar artifacts corresponding to the output of the current rule.
   * @return A nested set containing all of the source jar artifacts on which the current rule
   *     transitively depends.
   */
  public NestedSet<Artifact> collectTransitiveSourceJars(Iterable<Artifact> targetSrcJars)
      throws RuleErrorException {
    NestedSetBuilder<Artifact> builder =
        NestedSetBuilder.<Artifact>stableOrder().addAll(targetSrcJars);

    for (TransitiveInfoCollection dep : getDependencies()) {
      builder.addTransitive(JavaInfo.transitiveSourceJars(dep));
    }

    return builder.build();
  }

  /** Computes javacopts for the current rule. */
  private ImmutableList<String> computeJavacOpts(Collection<String> extraRuleJavacOpts)
      throws InterruptedException {
    ImmutableList.Builder<String> javacOpts =
        ImmutableList.<String>builder()
            .addAll(javaToolchain.getJavacOptions(ruleContext))
            .addAll(extraRuleJavacOpts);
    if (activePlugins
        .plugins()
        .processorClasses()
        .toSet()
        .contains("com.google.devtools.build.runfiles.AutoBazelRepositoryProcessor")) {
      javacOpts.add("-Abazel.repository=" + ruleContext.getRepository().getName());
    }
    return javacOpts
        .addAll(computePerPackageJavacOpts(ruleContext, javaToolchain))
        .addAll(addModuleJavacopts(ruleContext))
        .addAll(ruleContext.getExpander().withDataLocations().tokenized("javacopts"))
        .build();
  }

  private static ImmutableList<String> addModuleJavacopts(RuleContext ruleContext) {
    return JavaModuleFlagsProvider.create(ruleContext, Stream.empty()).toFlags();
  }

  /** Returns the per-package configured javacopts. */
  public static ImmutableList<String> computePerPackageJavacOpts(
      RuleContext ruleContext, JavaToolchainProvider toolchain) {
    // Do not use streams here as they create excessive garbage.
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (JavaPackageConfigurationProvider provider : toolchain.packageConfiguration()) {
      if (provider.matches(ruleContext.getLabel())) {
        result.addAll(provider.javacopts());
      }
    }
    return result.build();
  }

  /** Returns the per-package configured runfiles. */
  public static NestedSet<Artifact> computePerPackageData(
      RuleContext ruleContext, JavaToolchainProvider toolchain) {
    // Do not use streams here as they create excessive garbage.
    NestedSetBuilder<Artifact> data = NestedSetBuilder.naiveLinkOrder();
    for (JavaPackageConfigurationProvider provider : toolchain.packageConfiguration()) {
      if (provider.matches(ruleContext.getLabel())) {
        data.addTransitive(provider.data());
      }
    }
    return data.build();
  }

  public static PathFragment getHostJavaExecutable(RuleContext ruleContext) {
    return JavaRuntimeInfo.forHost(ruleContext).javaBinaryExecPathFragment();
  }

  public static PathFragment getHostJavaExecutable(JavaRuntimeInfo javaRuntime) {
    return javaRuntime.javaBinaryExecPathFragment();
  }

  public static PathFragment getJavaExecutable(RuleContext ruleContext) {
    return JavaRuntimeInfo.from(ruleContext).javaBinaryExecPathFragment();
  }

  /**
   * Returns the path of the java executable that the java stub should use.
   *
   * @param launcher if non-null, the cc_binary used to launch the Java Virtual Machine
   */
  public static String getJavaExecutableForStub(
      RuleContext ruleContext, @Nullable Artifact launcher) {
    Preconditions.checkState(ruleContext.getConfiguration().hasFragment(JavaConfiguration.class));
    PathFragment javaExecutable;
    JavaRuntimeInfo javaRuntime = JavaRuntimeInfo.from(ruleContext);

    if (launcher != null) {
      javaExecutable = launcher.getRootRelativePath();
    } else {
      javaExecutable = javaRuntime.javaBinaryRunfilesPathFragment();
    }

    if (!javaExecutable.isAbsolute()) {
      javaExecutable =
          PathFragment.create(ruleContext.getWorkspaceName()).getRelative(javaExecutable);
    }
    return javaExecutable.getPathString();
  }

  /**
   * Returns the shell command that computes `JAVABIN`. The command derives the JVM location from a
   * given Java executable path.
   */
  public static String getJavaBinSubstitutionFromJavaExecutable(
      RuleContext ruleContext, String javaExecutableStr) {
    PathFragment javaExecutable = PathFragment.create(javaExecutableStr);
    if (ruleContext.getConfiguration().runfilesEnabled()) {
      String prefix = "";
      if (!javaExecutable.isAbsolute()) {
        prefix = "${JAVA_RUNFILES}/";
      }
      return "JAVABIN=${JAVABIN:-" + prefix + javaExecutable.getPathString() + "}";
    } else {
      return "JAVABIN=${JAVABIN:-$(rlocation " + javaExecutable.getPathString() + ")}";
    }
  }

  /** Returns the string that the stub should use to determine the JVM binary (java) path */
  public static String getJavaBinSubstitution(
      RuleContext ruleContext, @Nullable Artifact launcher) {
    return getJavaBinSubstitutionFromJavaExecutable(
        ruleContext, getJavaExecutableForStub(ruleContext, launcher));
  }

  /**
   * Heuristically determines the name of the primary Java class for this executable, based on the
   * rule name and the "srcs" list.
   *
   * <p>(This is expected to be the class containing the "main" method for a java_binary, or a JUnit
   * Test class for a java_test.)
   *
   * @param sourceFiles the source files for this rule
   * @return a fully qualified Java class name, or null if none could be determined.
   */
  public static String determinePrimaryClass(
      RuleContext ruleContext, ImmutableList<Artifact> sourceFiles) {
    return determinePrimaryClass(ruleContext.getTarget(), sourceFiles);
  }

  private static String determinePrimaryClass(Target target, ImmutableList<Artifact> sourceFiles) {
    if (!sourceFiles.isEmpty()) {
      String mainSource = target.getName() + ".java";
      for (Artifact sourceFile : sourceFiles) {
        PathFragment path = sourceFile.getRootRelativePath();
        if (path.getBaseName().equals(mainSource)) {
          return JavaUtil.getJavaFullClassname(FileSystemUtils.removeExtension(path));
        }
      }
    }
    // Last resort: Use the name and package name of the target.
    // TODO(bazel-team): this should be fixed to use a source file from the dependencies to
    // determine the package of the Java class.
    return JavaUtil.getJavaFullClassname(Util.getWorkspaceRelativePath(target));
  }

  /**
   * Gets the value of the "jvm_flags" attribute combining it with the default options and expanding
   * any make variables and $(location) tags.
   */
  public static List<String> getJvmFlags(RuleContext ruleContext) throws InterruptedException {
    List<String> jvmFlags = new ArrayList<>();
    jvmFlags.addAll(ruleContext.getFragment(JavaConfiguration.class).getDefaultJvmFlags());
    jvmFlags.addAll(ruleContext.getExpander().withDataLocations().list("jvm_flags"));
    return jvmFlags;
  }

  private static List<TransitiveInfoCollection> getRuntimeDeps(RuleContext ruleContext) {
    // We need to check here because there are classes inheriting from this class that implement
    // rules that don't have this attribute.
    if (ruleContext.attributes().has("runtime_deps", BuildType.LABEL_LIST)) {
      return ImmutableList.copyOf(ruleContext.getPrerequisites("runtime_deps"));
    } else {
      return ImmutableList.of();
    }
  }

  /**
   * Initialize the common actions and build various collections of artifacts for the
   * initializationHook() methods of the subclasses.
   *
   * <p>Note that not all subclasses call this method.
   *
   * @return the processed attributes
   */
  public JavaTargetAttributes.Builder initCommon(
      Collection<Artifact> extraSrcs, Iterable<String> extraJavacOpts)
      throws RuleErrorException, InterruptedException {
    Preconditions.checkState(javacOpts == null);
    activePlugins = collectPlugins();
    javacOpts = computeJavacOpts(ImmutableList.copyOf(extraJavacOpts));

    JavaTargetAttributes.Builder javaTargetAttributes = new JavaTargetAttributes.Builder(semantics);
    javaCompilationHelper =
        new JavaCompilationHelper(ruleContext, semantics, javacOpts, javaTargetAttributes);

    processSrcs(javaTargetAttributes);
    javaTargetAttributes.addSourceArtifacts(sources);
    javaTargetAttributes.addSourceArtifacts(extraSrcs);
    processRuntimeDeps(javaTargetAttributes);

    if (disallowDepsWithoutSrcs(ruleContext.getRule().getRuleClass())
        && ruleContext.attributes().get("srcs", BuildType.LABEL_LIST).isEmpty()
        && !ruleContext.attributes().get("deps", BuildType.LABEL_LIST).isEmpty()) {
      ruleContext.attributeError("deps", "deps not allowed without srcs; move to runtime_deps?");
    }

    for (Artifact resource : semantics.collectResources(ruleContext)) {
      javaTargetAttributes.addResource(
          JavaHelper.getJavaResourcePath(semantics, ruleContext, resource), resource);
    }

    if (ruleContext.attributes().has("resource_jars", BuildType.LABEL_LIST)
        && ruleContext.getRule().isAttributeValueExplicitlySpecified("resource_jars")) {
      if (ruleContext.getFragment(JavaConfiguration.class).disallowResourceJars()) {
        ruleContext.attributeError(
            "resource_jars",
            "resource_jars are not supported; use java_import and deps or runtime_deps instead.");
      }
      javaTargetAttributes.addResourceJars(
          PrerequisiteArtifacts.nestedSet(ruleContext, "resource_jars"));
    }

    addPlugins(javaTargetAttributes);

    javaTargetAttributes.setTargetLabel(ruleContext.getLabel());

    return javaTargetAttributes;
  }

  private boolean disallowDepsWithoutSrcs(String ruleClass) {
    return ruleClass.equals("java_library")
        || ruleClass.equals("java_binary")
        || ruleClass.equals("java_test");
  }

  public ImmutableList<? extends TransitiveInfoCollection> targetsTreatedAsDeps(
      ClasspathType type) {
    return targetsTreatedAsDeps.get(type);
  }

  /** Returns the default dependencies for the given classpath context. */
  public static ImmutableList<TransitiveInfoCollection> defaultDeps(
      RuleContext ruleContext, JavaSemantics semantics, ClasspathType type) {
    return collectTargetsTreatedAsDeps(ruleContext, semantics, type);
  }

  private static ImmutableList<TransitiveInfoCollection> collectTargetsTreatedAsDeps(
      RuleContext ruleContext, JavaSemantics semantics, ClasspathType type) {
    ImmutableList.Builder<TransitiveInfoCollection> builder = new ImmutableList.Builder<>();

    if (!type.equals(ClasspathType.COMPILE_ONLY)) {
      builder.addAll(getRuntimeDeps(ruleContext));
      builder.addAll(getExports(ruleContext));
    }
    builder.addAll(ruleContext.getPrerequisites("deps"));

    semantics.collectTargetsTreatedAsDeps(ruleContext, builder, type);

    // Implicitly add dependency on java launcher cc_binary when --java_launcher= is enabled,
    // or when launcher attribute is specified in a build rule.
    TransitiveInfoCollection launcher = JavaHelper.launcherForTarget(semantics, ruleContext);
    if (launcher != null) {
      builder.add(launcher);
    }

    return builder.build();
  }

  public void addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      JavaInfo.Builder javaInfoBuilder,
      NestedSet<Artifact> filesToBuild,
      @Nullable Artifact classJar)
      throws RuleErrorException {
    addTransitiveInfoProviders(
        builder,
        javaInfoBuilder,
        filesToBuild,
        classJar,
        JAVA_COLLECTION_SPEC,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  public void addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      JavaInfo.Builder javaInfoBuilder,
      NestedSet<Artifact> filesToBuild,
      @Nullable Artifact classJar,
      InstrumentationSpec instrumentationSpec)
      throws RuleErrorException {
    addTransitiveInfoProviders(
        builder,
        javaInfoBuilder,
        filesToBuild,
        classJar,
        instrumentationSpec,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  public void addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      JavaInfo.Builder javaInfoBuilder,
      NestedSet<Artifact> filesToBuild,
      @Nullable Artifact classJar,
      NestedSet<Pair<String, String>> coverageEnvironment,
      NestedSet<Artifact> coverageSupportFiles)
      throws RuleErrorException {
    addTransitiveInfoProviders(
        builder,
        javaInfoBuilder,
        filesToBuild,
        classJar,
        JAVA_COLLECTION_SPEC,
        coverageEnvironment,
        coverageSupportFiles);
  }

  public void addTransitiveInfoProviders(
      RuleConfiguredTargetBuilder builder,
      JavaInfo.Builder javaInfoBuilder,
      NestedSet<Artifact> filesToBuild,
      @Nullable Artifact classJar,
      InstrumentationSpec instrumentationSpec,
      NestedSet<Pair<String, String>> coverageEnvironment,
      NestedSet<Artifact> coverageSupportFiles)
      throws RuleErrorException {

    JavaCompilationInfoProvider compilationInfoProvider = createCompilationInfoProvider();

    builder
        .addNativeDeclaredProvider(
            getInstrumentationFilesProvider(
                ruleContext,
                filesToBuild,
                instrumentationSpec,
                coverageEnvironment,
                coverageSupportFiles))
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, getFilesToCompile(classJar));

    javaInfoBuilder.javaCompilationInfo(compilationInfoProvider);

    addCcRelatedProviders(javaInfoBuilder);
  }

  /** Adds Cc related providers to a Java target. */
  private void addCcRelatedProviders(JavaInfo.Builder javaInfoBuilder) throws RuleErrorException {
    ImmutableList<? extends TransitiveInfoCollection> deps =
        targetsTreatedAsDeps(ClasspathType.BOTH);

    ImmutableList<CcInfo> ccInfos =
        Streams.concat(
                AnalysisUtils.getProviders(deps, CcInfo.PROVIDER).stream(),
                JavaInfo.ccInfos(deps).stream())
            .collect(toImmutableList());

    CcInfo mergedCcInfo = CcInfo.merge(ccInfos);

    javaInfoBuilder.javaCcInfo(JavaCcInfoProvider.create(mergedCcInfo));
  }

  private InstrumentedFilesInfo getInstrumentationFilesProvider(
      RuleContext ruleContext,
      NestedSet<Artifact> filesToBuild,
      InstrumentationSpec instrumentationSpec,
      NestedSet<Pair<String, String>> coverageEnvironment,
      NestedSet<Artifact> coverageSupportFiles) {

    return InstrumentedFilesCollector.collect(
        ruleContext,
        instrumentationSpec,
        InstrumentedFilesCollector.NO_METADATA_COLLECTOR,
        filesToBuild.toList(),
        coverageSupportFiles,
        coverageEnvironment,
        /* withBaselineCoverage= */ !TargetUtils.isTestRule(ruleContext.getTarget()),
        /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER));
  }

  public void addGenJarsProvider(
      RuleConfiguredTargetBuilder builder,
      JavaInfo.Builder javaInfoBuilder,
      @Nullable Artifact genClassJar,
      @Nullable Artifact genSourceJar)
      throws RuleErrorException {
    ImmutableList.Builder<JavaGenJarsProvider> transitive = ImmutableList.builder();
    for (TransitiveInfoCollection dep : getDependencies()) {
      transitive.add(JavaInfo.genJarsProvider(dep));
    }
    JavaGenJarsProvider genJarsProvider =
        JavaGenJarsProvider.create(
            javaCompilationHelper.usesAnnotationProcessing(),
            genClassJar,
            genSourceJar,
            activePlugins,
            transitive.build());

    javaInfoBuilder.javaGenJars(genJarsProvider);
  }

  /** Processes the sources of this target, adding them as messages or proper sources. */
  private void processSrcs(JavaTargetAttributes.Builder attributes) throws RuleErrorException {
    List<? extends TransitiveInfoCollection> srcs = ruleContext.getPrerequisites("srcs");
    for (TransitiveInfoCollection src : srcs) {
      ImmutableList<Artifact> messages = MessageBundleInfo.getMessages(src);
      if (messages != null) {
        attributes.addMessages(messages);
      }
    }
  }

  /** Processes the transitive runtime_deps of this target. */
  private void processRuntimeDeps(JavaTargetAttributes.Builder attributes)
      throws RuleErrorException {
    List<TransitiveInfoCollection> runtimeDepInfo = getRuntimeDeps(ruleContext);
    checkRuntimeDeps(ruleContext, runtimeDepInfo);
    JavaCompilationArgsProvider provider =
        JavaCompilationArgsProvider.legacyFromTargets(runtimeDepInfo);
    attributes.addRuntimeClassPathEntries(provider.getRuntimeJars());
  }

  /**
   * Adds information about the annotation processors that should be run for this java target to the
   * target attributes.
   */
  private void addPlugins(JavaTargetAttributes.Builder attributes) {
    addPlugins(attributes, activePlugins);
  }

  /**
   * Adds information about the annotation processors that should be run for this java target
   * retrieved from the given plugins to the target attributes.
   *
   * <p>In particular, the processor names/paths and the API generating processor names/paths are
   * added to the given attributes. Plugins having repetitive names/paths will be added only once.
   */
  public static void addPlugins(
      JavaTargetAttributes.Builder attributes, JavaPluginInfo activePlugins) {
    attributes.addPlugin(activePlugins);
  }

  private JavaPluginInfo collectPlugins() throws RuleErrorException {
    List<JavaPluginInfo> result = new ArrayList<>();
    Iterables.addAll(result, getDirectJavaPluginInfoForAttribute(ruleContext, ":java_plugins"));
    Iterables.addAll(result, getDirectJavaPluginInfoForAttribute(ruleContext, "plugins"));
    Iterables.addAll(result, getExportedJavaPluginInfoForAttribute(ruleContext, "deps"));
    return JavaPluginInfo.mergeWithoutJavaOutputs(result);
  }

  private static Iterable<JavaPluginInfo> getDirectJavaPluginInfoForAttribute(
      RuleContext ruleContext, String attribute) throws RuleErrorException {
    if (ruleContext.attributes().has(attribute, BuildType.LABEL_LIST)) {
      ImmutableList.Builder<JavaPluginInfo> builder = ImmutableList.builder();
      for (TransitiveInfoCollection target : ruleContext.getPrerequisites(attribute)) {
        JavaPluginInfo javaPluginInfo = target.get(JavaPluginInfo.PROVIDER);
        if (javaPluginInfo != null) {
          builder.add(javaPluginInfo);
        }
      }
      return builder.build();
    }
    return ImmutableList.of();
  }

  private static Iterable<JavaPluginInfo> getExportedJavaPluginInfoForAttribute(
      RuleContext ruleContext, String attribute) throws RuleErrorException {
    if (ruleContext.attributes().has(attribute, BuildType.LABEL_LIST)) {
      ImmutableList.Builder<JavaPluginInfo> builder = ImmutableList.builder();
      for (TransitiveInfoCollection target : ruleContext.getPrerequisites(attribute)) {
        JavaInfo javaInfo = target.get(JavaInfo.PROVIDER);
        if (javaInfo != null && javaInfo.getJavaPluginInfo() != null) {
          builder.add(javaInfo.getJavaPluginInfo());
        }
      }
      return builder.build();
    }
    return ImmutableList.of();
  }

  public static JavaPluginInfo getTransitivePlugins(RuleContext ruleContext)
      throws RuleErrorException {
    return JavaPluginInfo.mergeWithoutJavaOutputs(
        Iterables.concat(
            getDirectJavaPluginInfoForAttribute(ruleContext, "exported_plugins"),
            getExportedJavaPluginInfoForAttribute(ruleContext, "exports")));
  }

  public static Runfiles getRunfiles(
      RuleContext ruleContext,
      JavaSemantics semantics,
      JavaCompilationArtifacts javaArtifacts,
      boolean neverLink) {
    // The "neverlink" attribute is transitive, so we don't add any
    // runfiles from this target or its dependencies.
    if (neverLink) {
      return Runfiles.EMPTY;
    }
    Runfiles.Builder runfilesBuilder =
        new Runfiles.Builder(
                ruleContext.getWorkspaceName(),
                ruleContext.getConfiguration().legacyExternalRunfiles())
            .addArtifacts(javaArtifacts.getRuntimeJars());
    runfilesBuilder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);

    List<TransitiveInfoCollection> depsForRunfiles = new ArrayList<>();
    if (ruleContext.getRule().isAttrDefined("runtime_deps", BuildType.LABEL_LIST)) {
      depsForRunfiles.addAll(ruleContext.getPrerequisites("runtime_deps"));
    }
    if (ruleContext.getRule().isAttrDefined("exports", BuildType.LABEL_LIST)) {
      depsForRunfiles.addAll(ruleContext.getPrerequisites("exports"));
    }

    runfilesBuilder.addTargets(
        depsForRunfiles,
        RunfilesProvider.DEFAULT_RUNFILES,
        ruleContext.getConfiguration().alwaysIncludeFilesToBuildInData());

    TransitiveInfoCollection launcher = JavaHelper.launcherForTarget(semantics, ruleContext);
    if (launcher != null) {
      runfilesBuilder.addTarget(
          launcher,
          RunfilesProvider.DATA_RUNFILES,
          ruleContext.getConfiguration().alwaysIncludeFilesToBuildInData());
    }

    semantics.addRunfilesForLibrary(ruleContext, runfilesBuilder);
    return runfilesBuilder.build();
  }

  /** Gets all the deps. */
  public final List<? extends TransitiveInfoCollection> getDependencies() {
    return targetsTreatedAsDeps(ClasspathType.BOTH);
  }

  /**
   * Returns true if and only if this target has the neverlink attribute set to 1, or false if the
   * neverlink attribute does not exist (for example, on *_binary targets)
   *
   * @return the value of the neverlink attribute.
   */
  public static final boolean isNeverLink(RuleContext ruleContext) {
    return ruleContext.getRule().isAttrDefined("neverlink", Type.BOOLEAN)
        && ruleContext.attributes().get("neverlink", Type.BOOLEAN);
  }

  private static NestedSet<Artifact> getFilesToCompile(Artifact classJar) {
    if (classJar == null) {
      // Some subclasses don't produce jars
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return NestedSetBuilder.create(Order.STABLE_ORDER, classJar);
  }

  public ImmutableList<Artifact> getSrcsArtifacts() {
    return sources;
  }

  public ImmutableList<String> getJavacOpts() {
    return javacOpts;
  }

  public BootClassPathInfo getBootClasspath() {
    return classpathFragment.getBootClasspath();
  }

  public NestedSet<Artifact> getRuntimeClasspath() {
    return classpathFragment.getRuntimeClasspath();
  }

  public NestedSet<Artifact> getCompileTimeClasspath() {
    return classpathFragment.getCompileTimeClasspath();
  }

  public RuleContext getRuleContext() {
    return ruleContext;
  }

  private JavaCompilationInfoProvider createCompilationInfoProvider() {
    return new JavaCompilationInfoProvider.Builder()
        .setJavacOpts(javacOpts)
        .setBootClasspath(getBootClasspath())
        .setCompilationClasspath(getCompileTimeClasspath())
        .setRuntimeClasspath(getRuntimeClasspath())
        .build();
  }
}
