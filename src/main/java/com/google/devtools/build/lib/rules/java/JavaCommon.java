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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.Util;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcNativeLibraryInfo;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
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

  public JavaCommon(RuleContext ruleContext, JavaSemantics semantics) {
    this(
        ruleContext,
        semantics,
        ruleContext.getPrerequisiteArtifacts("srcs").list(),
        collectTargetsTreatedAsDeps(ruleContext, semantics, ClasspathType.COMPILE_ONLY),
        collectTargetsTreatedAsDeps(ruleContext, semantics, ClasspathType.RUNTIME_ONLY),
        collectTargetsTreatedAsDeps(ruleContext, semantics, ClasspathType.BOTH));
  }

  public JavaCommon(
      RuleContext ruleContext, JavaSemantics semantics, ImmutableList<Artifact> sources) {
    this(
        ruleContext,
        semantics,
        sources,
        collectTargetsTreatedAsDeps(ruleContext, semantics, ClasspathType.COMPILE_ONLY),
        collectTargetsTreatedAsDeps(ruleContext, semantics, ClasspathType.RUNTIME_ONLY),
        collectTargetsTreatedAsDeps(ruleContext, semantics, ClasspathType.BOTH));
  }

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

  /**
   * Creates an action to aggregate all metadata artifacts into a single
   * &lt;target_name&gt;_instrumented.jar file.
   */
  public static void createInstrumentedJarAction(
      RuleContext ruleContext,
      JavaSemantics semantics,
      List<Artifact> metadataArtifacts,
      Artifact instrumentedJar,
      String mainClass)
      throws InterruptedException {
    // In Jacoco's setup, metadata artifacts are real jars.
    new DeployArchiveBuilder(semantics, ruleContext)
        .setOutputJar(instrumentedJar)
        // We need to save the original mainClass because we're going to run inside CoverageRunner
        .setJavaStartClass(mainClass)
        .setAttributes(new JavaTargetAttributes.Builder(semantics).build())
        .addRuntimeJars(ImmutableList.copyOf(metadataArtifacts))
        .setCompression(DeployArchiveBuilder.Compression.UNCOMPRESSED)
        .build();
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
   * Creates the java.library.path from a list of the native libraries. Concatenates the parent
   * directories of the shared libraries into a Java search path. Each relative path entry is
   * prepended with "${JAVA_RUNFILES}/" so it can be resolved at runtime.
   *
   * @param sharedLibraries a collection of native libraries to create the java library path from
   * @return a String containing the ":" separated java library path
   */
  public static String javaLibraryPath(Collection<Artifact> sharedLibraries, String runfilePrefix) {
    StringBuilder buffer = new StringBuilder();
    Set<PathFragment> entries = new HashSet<>();
    for (Artifact sharedLibrary : sharedLibraries) {
      PathFragment entry = sharedLibrary.getRootRelativePath().getParentDirectory();
      if (entries.add(entry)) {
        if (buffer.length() > 0) {
          buffer.append(':');
        }
        buffer
            .append("${JAVA_RUNFILES}/")
            .append(runfilePrefix)
            .append("/")
            .append(entry.getPathString());
      }
    }
    return buffer.toString();
  }

  /**
   * Collects Java compilation arguments for this target.
   *
   * @param isNeverLink Whether the target has the 'neverlink' attr.
   * @param srcLessDepsExport If srcs is omitted, deps are exported (deprecated behaviour for
   *     android_library only)
   */
  public JavaCompilationArgsProvider collectJavaCompilationArgs(
      boolean isNeverLink, boolean srcLessDepsExport) {
    return collectJavaCompilationArgs(
        /* isNeverLink= */ isNeverLink,
        /* srcLessDepsExport= */ srcLessDepsExport,
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
      boolean srcLessDepsExport,
      JavaCompilationArtifacts compilationArtifacts,
      List<JavaCompilationArgsProvider> deps,
      List<JavaCompilationArgsProvider> runtimeDeps,
      List<JavaCompilationArgsProvider> exports) {
    ClasspathType type = isNeverLink ? ClasspathType.COMPILE_ONLY : ClasspathType.BOTH;
    JavaCompilationArgsProvider.Builder builder =
        JavaCompilationArgsProvider.builder().merge(compilationArtifacts, isNeverLink);
    exports.forEach(export -> builder.addExports(export, type));
    if (srcLessDepsExport) {
      deps.forEach(dep -> builder.addExports(dep, type));
    } else {
      deps.forEach(dep -> builder.addDeps(dep, type));
    }
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
      RuleContext ruleContext, List<TransitiveInfoCollection> runtimeDepInfo) {
    for (TransitiveInfoCollection c : runtimeDepInfo) {
      JavaInfo javaInfo = (JavaInfo) c.get(JavaInfo.PROVIDER.getKey());
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
   * Returns transitive Java native libraries.
   *
   * @see JavaNativeLibraryInfo
   */
  protected NestedSet<LibraryToLink> collectTransitiveJavaNativeLibraries() {
    NativeLibraryNestedSetBuilder builder = new NativeLibraryNestedSetBuilder();
    builder.addJavaTargets(targetsTreatedAsDeps(ClasspathType.BOTH));

    if (ruleContext.getRule().isAttrDefined("data", BuildType.LABEL_LIST)) {
      builder.addJavaTargets(ruleContext.getPrerequisites("data"));
    }
    return builder.build();
  }

  /**
   * Collects transitive source jars for the current rule.
   *
   * @param targetSrcJars The source jar artifacts corresponding to the output of the current rule.
   * @return A nested set containing all of the source jar artifacts on which the current rule
   *     transitively depends.
   */
  public NestedSet<Artifact> collectTransitiveSourceJars(Artifact... targetSrcJars) {
    return collectTransitiveSourceJars(ImmutableList.copyOf(targetSrcJars));
  }

  /**
   * Collects transitive source jars for the current rule.
   *
   * @param targetSrcJars The source jar artifacts corresponding to the output of the current rule.
   * @return A nested set containing all of the source jar artifacts on which the current rule
   *     transitively depends.
   */
  public NestedSet<Artifact> collectTransitiveSourceJars(Iterable<Artifact> targetSrcJars) {
    NestedSetBuilder<Artifact> builder =
        NestedSetBuilder.<Artifact>stableOrder().addAll(targetSrcJars);

    for (JavaSourceJarsProvider sourceJarsProvider :
        JavaInfo.getProvidersFromListOfTargets(JavaSourceJarsProvider.class, getDependencies())) {
      builder.addTransitive(sourceJarsProvider.getTransitiveSourceJars());
    }

    return builder.build();
  }

  /** Collects labels of targets and artifacts reached transitively via the "exports" attribute. */
  protected JavaExportsProvider collectTransitiveExports() {
    NestedSetBuilder<Label> builder = NestedSetBuilder.stableOrder();
    List<TransitiveInfoCollection> currentRuleExports = getExports(ruleContext);

    builder.addAll(Iterables.transform(currentRuleExports, TransitiveInfoCollection::getLabel));

    for (TransitiveInfoCollection dep : currentRuleExports) {
      JavaExportsProvider exportsProvider = JavaInfo.getProvider(JavaExportsProvider.class, dep);

      if (exportsProvider != null) {
        builder.addTransitive(exportsProvider.getTransitiveExports());
      }
    }

    return new JavaExportsProvider(builder.build());
  }

  public final void initializeJavacOpts() {
    Preconditions.checkState(javacOpts == null);
    javacOpts = computeJavacOpts(getCompatibleJavacOptions());
  }

  /** Computes javacopts for the current rule. */
  private ImmutableList<String> computeJavacOpts(Collection<String> extraRuleJavacOpts) {
    return ImmutableList.<String>builder()
        .addAll(javaToolchain.getJavacOptions(ruleContext))
        .addAll(extraRuleJavacOpts)
        .addAll(computePerPackageJavacOpts(ruleContext, javaToolchain))
        .addAll(ruleContext.getExpander().withDataLocations().tokenized("javacopts"))
        .build();
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
  public static List<String> getJvmFlags(RuleContext ruleContext) {
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

  public JavaTargetAttributes.Builder initCommon() {
    return initCommon(ImmutableList.of(), getCompatibleJavacOptions());
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
      Collection<Artifact> extraSrcs, Iterable<String> extraJavacOpts) {
    Preconditions.checkState(javacOpts == null);
    javacOpts = computeJavacOpts(ImmutableList.copyOf(extraJavacOpts));
    activePlugins = collectPlugins();

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

  private ImmutableList<String> getCompatibleJavacOptions() {
    return semantics.getCompatibleJavacOptions(ruleContext, javaToolchain);
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
      @Nullable Artifact classJar) {
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
      InstrumentationSpec instrumentationSpec) {
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
      NestedSet<Artifact> coverageSupportFiles) {
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
      NestedSet<Artifact> coverageSupportFiles) {

    JavaCompilationInfoProvider compilationInfoProvider = createCompilationInfoProvider();
    JavaExportsProvider exportsProvider = collectTransitiveExports();

    builder
        .addNativeDeclaredProvider(
            getInstrumentationFilesProvider(
                ruleContext,
                filesToBuild,
                instrumentationSpec,
                coverageEnvironment,
                coverageSupportFiles))
        .addOutputGroup(OutputGroupInfo.FILES_TO_COMPILE, getFilesToCompile(classJar));

    javaInfoBuilder.addProvider(JavaExportsProvider.class, exportsProvider);
    javaInfoBuilder.addProvider(JavaCompilationInfoProvider.class, compilationInfoProvider);

    addCcRelatedProviders(builder, javaInfoBuilder);
  }

  /** Adds Cc related providers to a Java target. */
  private void addCcRelatedProviders(
      RuleConfiguredTargetBuilder ruleBuilder, JavaInfo.Builder javaInfoBuilder) {
    Iterable<? extends TransitiveInfoCollection> deps = targetsTreatedAsDeps(ClasspathType.BOTH);


    ImmutableList<CcInfo> ccInfos =
        Streams.concat(
                AnalysisUtils.getProviders(deps, CcInfo.PROVIDER).stream(),
                AnalysisUtils.getProviders(deps, JavaCcLinkParamsProvider.PROVIDER).stream()
                    .map(JavaCcLinkParamsProvider::getCcInfo),
                JavaInfo.getProvidersFromListOfTargets(JavaCcInfoProvider.class, deps).stream()
                    .map(JavaCcInfoProvider::getCcInfo))
            .collect(toImmutableList());

    CcInfo mergedCcInfo = CcInfo.merge(ccInfos);

    // Collect library paths from all attributes (including data)
    Iterable<? extends TransitiveInfoCollection> data;
    if (ruleContext.getRule().isAttrDefined("data", BuildType.LABEL_LIST)) {
      data = ruleContext.getPrerequisites("data");
    } else {
      data = ImmutableList.of();
    }
    CcNativeLibraryInfo mergedCcNativeLibraryInfo =
        CcNativeLibraryInfo.merge(
            Streams.concat(
                    Stream.of(mergedCcInfo.getCcNativeLibraryInfo()),
                    AnalysisUtils.getProviders(
                            Iterables.concat(deps, data), JavaNativeLibraryInfo.PROVIDER)
                        .stream()
                        .map(JavaNativeLibraryInfo::getTransitiveJavaNativeLibraries)
                        .map(CcNativeLibraryInfo::new),
                    JavaInfo.getProvidersFromListOfTargets(JavaCcInfoProvider.class, data).stream()
                        .map(JavaCcInfoProvider::getCcInfo)
                        .map(CcInfo::getCcNativeLibraryInfo),
                    AnalysisUtils.getProviders(data, CcInfo.PROVIDER).stream()
                        .map(CcInfo::getCcNativeLibraryInfo))
                .collect(toImmutableList()));

    CcInfo filteredCcInfo =
        CcInfo.builder()
            .setCcLinkingContext(mergedCcInfo.getCcLinkingContext())
            .setCcNativeLibraryInfo(mergedCcNativeLibraryInfo)
            .build();

    if (ruleContext
        .getFragment(JavaConfiguration.class)
        .experimentalPublishJavaCcLinkParamsInfo()) {
      ruleBuilder.addNativeDeclaredProvider(new JavaCcLinkParamsProvider(filteredCcInfo));
    }
    javaInfoBuilder.addProvider(JavaCcInfoProvider.class, new JavaCcInfoProvider(filteredCcInfo));
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
      @Nullable Artifact genSourceJar) {
    JavaGenJarsProvider genJarsProvider =
        JavaGenJarsProvider.create(
            javaCompilationHelper.usesAnnotationProcessing(),
            genClassJar,
            genSourceJar,
            activePlugins,
            getDependencies(JavaGenJarsProvider.class));

    builder.addProvider(JavaGenJarsProvider.class, genJarsProvider);

    javaInfoBuilder.addProvider(JavaGenJarsProvider.class, genJarsProvider);
  }

  /** Processes the sources of this target, adding them as messages or proper sources. */
  private void processSrcs(JavaTargetAttributes.Builder attributes) {
    List<? extends TransitiveInfoCollection> srcs = ruleContext.getPrerequisites("srcs");
    for (TransitiveInfoCollection src : srcs) {
      ImmutableList<Artifact> messages = MessageBundleInfo.getMessages(src);
      if (messages != null) {
        attributes.addMessages(messages);
      }
    }
  }

  /** Processes the transitive runtime_deps of this target. */
  private void processRuntimeDeps(JavaTargetAttributes.Builder attributes) {
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

  private JavaPluginInfo collectPlugins() {
    List<JavaPluginInfo> result = new ArrayList<>();
    Iterables.addAll(result, getJavaPluginInfoForAttribute(ruleContext, ":java_plugins"));
    Iterables.addAll(result, getJavaPluginInfoForAttribute(ruleContext, "plugins"));
    Iterables.addAll(result, getJavaPluginInfoForAttribute(ruleContext, "deps"));
    return JavaPluginInfo.merge(result);
  }

  private static Iterable<JavaPluginInfo> getJavaPluginInfoForAttribute(
      RuleContext ruleContext, String attribute) {
    if (ruleContext.attributes().has(attribute, BuildType.LABEL_LIST)) {
      return ruleContext.getPrerequisites(attribute).stream()
          .map(JavaInfo::getJavaInfo)
          .filter(Objects::nonNull)
          .map(JavaInfo::getJavaPluginInfo)
          .filter(Objects::nonNull)
          .collect(toImmutableList());
    }
    return ImmutableList.of();
  }

  JavaPluginInfo createJavaPluginInfo(RuleContext ruleContext) {
    NestedSet<String> processorClasses =
        NestedSetBuilder.wrap(Order.NAIVE_LINK_ORDER, getProcessorClasses(ruleContext));
    NestedSet<Artifact> processorClasspath = getRuntimeClasspath();
    FileProvider dataProvider = ruleContext.getPrerequisite("data", FileProvider.class);
    NestedSet<Artifact> data =
        dataProvider != null
            ? dataProvider.getFilesToBuild()
            : NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    return JavaPluginInfo.create(
        JavaPluginData.create(processorClasses, processorClasspath, data),
        ruleContext.attributes().get("generates_api", Type.BOOLEAN));
  }

  /**
   * Returns the class that should be passed to javac in order to run the annotation processor this
   * class represents.
   */
  private static ImmutableSet<String> getProcessorClasses(RuleContext ruleContext) {
    return ruleContext.getRule().isAttributeValueExplicitlySpecified("processor_class")
        ? ImmutableSet.of(ruleContext.attributes().get("processor_class", Type.STRING))
        : ImmutableSet.of();
  }

  public static JavaPluginInfo getTransitivePlugins(RuleContext ruleContext) {
    return JavaPluginInfo.merge(
        Iterables.concat(
            getJavaPluginInfoForAttribute(ruleContext, "exported_plugins"),
            getJavaPluginInfoForAttribute(ruleContext, "exports")));
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

    runfilesBuilder.addTargets(depsForRunfiles, RunfilesProvider.DEFAULT_RUNFILES);

    TransitiveInfoCollection launcher = JavaHelper.launcherForTarget(semantics, ruleContext);
    if (launcher != null) {
      runfilesBuilder.addTarget(launcher, RunfilesProvider.DATA_RUNFILES);
    }

    semantics.addRunfilesForLibrary(ruleContext, runfilesBuilder);
    return runfilesBuilder.build();
  }

  /** Gets all the deps. */
  public final List<? extends TransitiveInfoCollection> getDependencies() {
    return targetsTreatedAsDeps(ClasspathType.BOTH);
  }

  /** Gets all the deps that implement a particular provider. */
  public final <P extends TransitiveInfoProvider> List<P> getDependencies(Class<P> provider) {
    return JavaInfo.getProvidersFromListOfTargets(provider, getDependencies());
  }

  /**
   * Returns a list of the current target's runtime jars and the first two levels of its direct
   * dependencies.
   *
   * <p>This method is meant to aid the persistent test runner, which aims at avoiding loading all
   * classes on the classpath for each test run. To that extent this method computes a small jars
   * set of the most likely to be changed classes when writing code for a test. Their classes should
   * be loaded in a separate classloader by the persistent test runner.
   */
  public ImmutableSet<Artifact> getDirectRuntimeClasspath() {
    ImmutableSet.Builder<Artifact> directDeps = new ImmutableSet.Builder<>();
    directDeps.addAll(javaArtifacts.getRuntimeJars());
    for (TransitiveInfoCollection dep : targetsTreatedAsDeps(ClasspathType.RUNTIME_ONLY)) {
      JavaInfo javaInfo = JavaInfo.getJavaInfo(dep);
      if (javaInfo != null) {
        directDeps.addAll(javaInfo.getDirectRuntimeJars());
      }
    }
    return directDeps.build();
  }

  /**
   * Return the runtime jars of the transitive closure of the target, excluding the first level of
   * dependencies and the current target itself.
   *
   * <p>This particular set of jars is used by the persistent test runner, to create a classloader
   * for the transitive dependencies. The target itself and its direct dependencies are loaded into
   * a different classloader.
   */
  public NestedSet<Artifact> getRuntimeClasspathExcludingDirect() {
    NestedSetBuilder<Artifact> classpath = new NestedSetBuilder<>(Order.STABLE_ORDER);
    targetsTreatedAsDeps(ClasspathType.RUNTIME_ONLY).stream()
        .map(JavaInfo::getJavaInfo)
        .filter(Objects::nonNull)
        .forEach(j -> classpath.addTransitive(j.getTransitiveOnlyRuntimeJars()));
    return classpath.build();
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
