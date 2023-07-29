// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.StandardSystemProperty.JAVA_SPECIFICATION_VERSION;
import static com.google.devtools.build.lib.rules.java.JavaStarlarkCommon.checkPrivateAccess;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.PackageSpecificationProvider;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.rules.java.JavaPluginInfo.JavaPluginData;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaToolchainStarlarkApiProviderApi;
import java.util.Iterator;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** Information about the JDK used by the <code>java_*</code> rules. */
@Immutable
public final class JavaToolchainProvider extends NativeInfo
    implements JavaToolchainStarlarkApiProviderApi {

  public static final BuiltinProvider<JavaToolchainProvider> PROVIDER =
      new BuiltinProvider<JavaToolchainProvider>(
          "JavaToolchainInfo", JavaToolchainProvider.class) {};

  /** Returns the Java Toolchain associated with the rule being analyzed or {@code null}. */
  public static JavaToolchainProvider from(RuleContext ruleContext) {
    ToolchainInfo toolchainInfo =
        ruleContext.getToolchainInfo(
            ruleContext
                .getPrerequisite(JavaRuleClasses.JAVA_TOOLCHAIN_TYPE_ATTRIBUTE_NAME)
                .getLabel());
    return from(toolchainInfo, ruleContext);
  }

  @VisibleForTesting
  public static JavaToolchainProvider from(ProviderCollection collection) {
    ToolchainInfo toolchainInfo = collection.get(ToolchainInfo.PROVIDER);
    return from(toolchainInfo, null);
  }

  @Nullable
  private static JavaToolchainProvider from(
      ToolchainInfo toolchainInfo, @Nullable RuleErrorConsumer errorConsumer) {
    if (toolchainInfo != null) {
      try {
        JavaToolchainProvider provider = (JavaToolchainProvider) toolchainInfo.getValue("java");
        if (provider != null) {
          return provider;
        }
      } catch (EvalException e) {
        if (errorConsumer != null) {
          errorConsumer.ruleError(
              String.format("There was an error reading the Java toolchain: %s", e));
        }
      }
    }
    if (errorConsumer != null) {
      errorConsumer.ruleError("The selected Java toolchain is not a JavaToolchainProvider");
    }
    return null;
  }

  public static JavaToolchainProvider create(
      Label label,
      ImmutableList<String> javacOptions,
      NestedSet<String> jvmOptions,
      boolean javacSupportsWorkers,
      boolean javacSupportsMultiplexWorkers,
      boolean javacSupportsWorkerCancellation,
      BootClassPathInfo bootclasspath,
      NestedSet<Artifact> tools,
      JavaToolchainTool javaBuilder,
      @Nullable JavaToolchainTool headerCompiler,
      @Nullable JavaToolchainTool headerCompilerDirect,
      @Nullable AndroidLintTool androidLint,
      JspecifyInfo jspecifyInfo,
      @Nullable JavaToolchainTool bytecodeOptimizer,
      ImmutableList<Artifact> localJavaOptimizationConfiguration,
      ImmutableSet<String> headerCompilerBuiltinProcessors,
      ImmutableSet<String> reducedClasspathIncompatibleProcessors,
      boolean forciblyDisableHeaderCompilation,
      FilesToRunProvider singleJar,
      @Nullable FilesToRunProvider oneVersion,
      @Nullable Artifact oneVersionAllowlist,
      @Nullable Artifact oneVersionAllowlistForTests,
      Artifact genClass,
      @Nullable Artifact depsChecker,
      @Nullable Artifact timezoneData,
      FilesToRunProvider ijar,
      ImmutableListMultimap<String, String> compatibleJavacOptions,
      ImmutableList<JavaPackageConfigurationProvider> packageConfiguration,
      FilesToRunProvider jacocoRunner,
      FilesToRunProvider proguardAllowlister,
      JavaSemantics javaSemantics,
      JavaRuntimeInfo javaRuntime) {
    return new JavaToolchainProvider(
        label,
        bootclasspath,
        tools,
        javaBuilder,
        headerCompiler,
        headerCompilerDirect,
        androidLint,
        jspecifyInfo,
        bytecodeOptimizer,
        localJavaOptimizationConfiguration,
        headerCompilerBuiltinProcessors,
        reducedClasspathIncompatibleProcessors,
        forciblyDisableHeaderCompilation,
        singleJar,
        oneVersion,
        oneVersionAllowlist,
        oneVersionAllowlistForTests,
        genClass,
        depsChecker,
        timezoneData,
        ijar,
        compatibleJavacOptions,
        javacOptions,
        jvmOptions,
        javacSupportsWorkers,
        javacSupportsMultiplexWorkers,
        javacSupportsWorkerCancellation,
        packageConfiguration,
        jacocoRunner,
        proguardAllowlister,
        javaSemantics,
        javaRuntime);
  }

  private final Label label;
  private final BootClassPathInfo bootclasspath;
  private final NestedSet<Artifact> tools;
  private final JavaToolchainTool javaBuilder;
  @Nullable private final JavaToolchainTool headerCompiler;
  @Nullable private final JavaToolchainTool headerCompilerDirect;
  @Nullable private final AndroidLintTool androidLint;
  @Nullable private final JspecifyInfo jspecifyInfo;
  @Nullable private final JavaToolchainTool bytecodeOptimizer;
  private final ImmutableList<Artifact> localJavaOptimizationConfiguration;
  private final ImmutableSet<String> headerCompilerBuiltinProcessors;
  private final ImmutableSet<String> reducedClasspathIncompatibleProcessors;
  private final boolean forciblyDisableHeaderCompilation;
  private final FilesToRunProvider singleJar;
  @Nullable private final FilesToRunProvider oneVersion;
  @Nullable private final Artifact oneVersionAllowlist;
  @Nullable private final Artifact oneVersionAllowlistForTests;
  private final Artifact genClass;
  @Nullable private final Artifact depsChecker;
  @Nullable private final Artifact timezoneData;
  private final FilesToRunProvider ijar;
  private final ImmutableListMultimap<String, String> compatibleJavacOptions;
  private final ImmutableList<String> javacOptions;
  private final NestedSet<String> jvmOptions;
  private final boolean javacSupportsWorkers;
  private final boolean javacSupportsMultiplexWorkers;
  private final boolean javacSupportsWorkerCancellation;
  private final ImmutableList<JavaPackageConfigurationProvider> packageConfiguration;
  private final FilesToRunProvider jacocoRunner;
  private final FilesToRunProvider proguardAllowlister;
  private final JavaSemantics javaSemantics;
  private final JavaRuntimeInfo javaRuntime;

  private JavaToolchainProvider(
      Label label,
      BootClassPathInfo bootclasspath,
      NestedSet<Artifact> tools,
      JavaToolchainTool javaBuilder,
      @Nullable JavaToolchainTool headerCompiler,
      @Nullable JavaToolchainTool headerCompilerDirect,
      @Nullable AndroidLintTool androidLint,
      @Nullable JspecifyInfo jspecifyInfo,
      @Nullable JavaToolchainTool bytecodeOptimizer,
      ImmutableList<Artifact> localJavaOptimizationConfiguration,
      ImmutableSet<String> headerCompilerBuiltinProcessors,
      ImmutableSet<String> reducedClasspathIncompatibleProcessors,
      boolean forciblyDisableHeaderCompilation,
      FilesToRunProvider singleJar,
      @Nullable FilesToRunProvider oneVersion,
      @Nullable Artifact oneVersionAllowlist,
      @Nullable Artifact oneVersionAllowlistForTests,
      Artifact genClass,
      @Nullable Artifact depsChecker,
      @Nullable Artifact timezoneData,
      FilesToRunProvider ijar,
      ImmutableListMultimap<String, String> compatibleJavacOptions,
      ImmutableList<String> javacOptions,
      NestedSet<String> jvmOptions,
      boolean javacSupportsWorkers,
      boolean javacSupportsMultiplexWorkers,
      boolean javacSupportsWorkerCancellation,
      ImmutableList<JavaPackageConfigurationProvider> packageConfiguration,
      FilesToRunProvider jacocoRunner,
      FilesToRunProvider proguardAllowlister,
      JavaSemantics javaSemantics,
      JavaRuntimeInfo javaRuntime) {

    this.label = label;
    this.bootclasspath = bootclasspath;
    this.tools = tools;
    this.javaBuilder = javaBuilder;
    this.headerCompiler = headerCompiler;
    this.headerCompilerDirect = headerCompilerDirect;
    this.androidLint = androidLint;
    this.jspecifyInfo = jspecifyInfo;
    this.bytecodeOptimizer = bytecodeOptimizer;
    this.localJavaOptimizationConfiguration = localJavaOptimizationConfiguration;
    this.headerCompilerBuiltinProcessors = headerCompilerBuiltinProcessors;
    this.reducedClasspathIncompatibleProcessors = reducedClasspathIncompatibleProcessors;
    this.forciblyDisableHeaderCompilation = forciblyDisableHeaderCompilation;
    this.singleJar = singleJar;
    this.oneVersion = oneVersion;
    this.oneVersionAllowlist = oneVersionAllowlist;
    this.oneVersionAllowlistForTests = oneVersionAllowlistForTests;
    this.genClass = genClass;
    this.depsChecker = depsChecker;
    this.timezoneData = timezoneData;
    this.ijar = ijar;
    this.compatibleJavacOptions = compatibleJavacOptions;
    this.javacOptions = javacOptions;
    this.jvmOptions = jvmOptions;
    this.javacSupportsWorkers = javacSupportsWorkers;
    this.javacSupportsMultiplexWorkers = javacSupportsMultiplexWorkers;
    this.javacSupportsWorkerCancellation = javacSupportsWorkerCancellation;
    this.packageConfiguration = packageConfiguration;
    this.jacocoRunner = jacocoRunner;
    this.proguardAllowlister = proguardAllowlister;
    this.javaSemantics = javaSemantics;
    this.javaRuntime = javaRuntime;
  }

  /** Returns the label for this {@code java_toolchain}. */
  @Override
  public Label getToolchainLabel() {
    return label;
  }

  /** Returns the target Java bootclasspath. */
  public BootClassPathInfo getBootclasspath() {
    return bootclasspath;
  }

  /** Returns the {@link Artifact}s of compilation tools. */
  public NestedSet<Artifact> getTools() {
    return tools;
  }

  /** Returns the {@link JavaToolchainTool} for JavaBuilder */
  public JavaToolchainTool getJavaBuilder() {
    return javaBuilder;
  }

  /** Returns the {@link JavaToolchainTool} for the header compiler */
  @Nullable
  public JavaToolchainTool getHeaderCompiler() {
    return headerCompiler;
  }

  /**
   * Returns the {@link FilesToRunProvider} of the Header Compiler deploy jar for direct-classpath,
   * non-annotation processing actions.
   */
  @Nullable
  public JavaToolchainTool getHeaderCompilerDirect() {
    return headerCompilerDirect;
  }

  @Nullable
  public AndroidLintTool getAndroidLint() {
    return androidLint;
  }

  @Nullable
  public JspecifyInfo jspecifyInfo() {
    return jspecifyInfo;
  }

  @Nullable
  public JavaToolchainTool getBytecodeOptimizer() {
    return bytecodeOptimizer;
  }

  public ImmutableList<Artifact> getLocalJavaOptimizationConfiguration() {
    return localJavaOptimizationConfiguration;
  }

  /** Returns class names of annotation processors that are built in to the header compiler. */
  public ImmutableSet<String> getHeaderCompilerBuiltinProcessors() {
    return headerCompilerBuiltinProcessors;
  }

  public ImmutableSet<String> getReducedClasspathIncompatibleProcessors() {
    return reducedClasspathIncompatibleProcessors;
  }

  /**
   * Returns {@code true} if header compilation should be forcibly disabled, overriding
   * --java_header_compilation.
   */
  public boolean getForciblyDisableHeaderCompilation() {
    return forciblyDisableHeaderCompilation;
  }

  @Override
  public boolean getForciblyDisableHeaderCompilationStarlark(StarlarkThread thread)
      throws EvalException {
    checkPrivateAccess(thread);
    return getForciblyDisableHeaderCompilation();
  }

  @Override
  public boolean hasHeaderCompiler(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return getHeaderCompiler() != null;
  }

  @Override
  public boolean hasHeaderCompilerDirect(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return getHeaderCompilerDirect() != null;
  }

  /** Returns the {@link FilesToRunProvider} of the SingleJar tool. */
  @Override
  public FilesToRunProvider getSingleJar() {
    return singleJar;
  }

  /**
   * Return the {@link FilesToRunProvider} of the tool that enforces one-version compliance of Java
   * binaries.
   */
  @Override
  @Nullable
  public FilesToRunProvider getOneVersionBinary() {
    return oneVersion;
  }

  /** Return the {@link Artifact} of the allowlist used by the one-version compliance checker. */
  @Nullable
  @Override
  public Artifact getOneVersionAllowlist() {
    return oneVersionAllowlist;
  }

  /**
   * Return the {@link Artifact} of the one-version allowlist for tests used by the one-version
   * compliance checker.
   */
  @Nullable
  public Artifact oneVersionAllowlistForTests() {
    return oneVersionAllowlistForTests;
  }

  @Override
  @Nullable
  public Artifact getOneVersionAllowlistForTests(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return oneVersionAllowlistForTests();
  }

  /** Returns the {@link Artifact} of the GenClass deploy jar */
  public Artifact getGenClass() {
    return genClass;
  }

  /** Returns the {@link Artifact} of the ImportDepsChecker deploy jar */
  @Nullable
  public Artifact depsChecker() {
    return depsChecker;
  }

  @Override
  public Artifact getDepsCheckerForStarlark(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return depsChecker();
  }

  /**
   * Returns the {@link Artifact} of the latest timezone data resource jar that can be loaded by
   * Java 8 binaries.
   */
  @Nullable
  public Artifact getTimezoneData() {
    return timezoneData;
  }

  /** Returns the ijar executable */
  @Override
  public FilesToRunProvider getIjar() {
    return ijar;
  }

  ImmutableListMultimap<String, String> getCompatibleJavacOptions() {
    return compatibleJavacOptions;
  }

  /** Returns the map of target environment-specific javacopts. */
  public ImmutableList<String> getCompatibleJavacOptions(String key) {
    return getCompatibleJavacOptions().get(key);
  }

  /** Returns the list of default options for the java compiler. */
  public ImmutableList<String> getJavacOptions(RuleContext ruleContext) {
    ImmutableList.Builder<String> result = ImmutableList.<String>builder().addAll(javacOptions);
    if (ruleContext != null) {
      // TODO(b/78512644): require ruleContext to be non-null after java_common.default_javac_opts
      // is turned down
      result.addAll(ruleContext.getFragment(JavaConfiguration.class).getDefaultJavacFlags());
    }
    return result.build();
  }

  /**
   * Returns the NestedSet of default options for the JVM running the java compiler and associated
   * tools.
   */
  public NestedSet<String> getJvmOptions() {
    return jvmOptions;
  }

  /** Returns whether JavaBuilders supports running as a persistent worker or not. */
  public boolean getJavacSupportsWorkers() {
    return javacSupportsWorkers;
  }

  /** Returns whether JavaBuilders supports running persistent workers in multiplex mode */
  public boolean getJavacSupportsMultiplexWorkers() {
    return javacSupportsMultiplexWorkers;
  }

  /** Returns whether JavaBuilders supports running persistent workers with cancellation */
  public boolean getJavacSupportsWorkerCancellation() {
    return javacSupportsWorkerCancellation;
  }

  /** Returns the global {@code java_package_configuration} data. */
  public ImmutableList<JavaPackageConfigurationProvider> packageConfiguration() {
    return packageConfiguration;
  }

  @Override
  public FilesToRunProvider getJacocoRunner() {
    return jacocoRunner;
  }

  @Override
  public FilesToRunProvider getProguardAllowlister() {
    return proguardAllowlister;
  }

  @Override
  public StarlarkList<JavaPackageConfigurationProvider> getPackageConfigurationStarlark(
      StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return StarlarkList.immutableCopyOf(packageConfiguration);
  }

  public JavaSemantics getJavaSemantics() {
    return javaSemantics;
  }

  @Override
  public JavaRuntimeInfo getJavaRuntime() {
    return javaRuntime;
  }

  @Override
  @Nullable
  public AndroidLintTool stalarkAndroidLinter(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return getAndroidLint();
  }

  /** Returns the input Java language level */
  // TODO(cushon): remove this API; it bakes a deprecated detail of the javac API into Bazel
  @Override
  public String getSourceVersion() {
    Iterator<String> it = javacOptions.iterator();
    while (it.hasNext()) {
      if (it.next().equals("-source") && it.hasNext()) {
        return it.next();
      }
    }
    return JAVA_SPECIFICATION_VERSION.value();
  }

  /** Returns the target Java language level */
  // TODO(cushon): remove this API; it bakes a deprecated detail of the javac API into Bazel
  @Override
  public String getTargetVersion() {
    Iterator<String> it = javacOptions.iterator();
    while (it.hasNext()) {
      if (it.next().equals("-target") && it.hasNext()) {
        return it.next();
      }
    }
    return JAVA_SPECIFICATION_VERSION.value();
  }

  @Override
  public Depset getStarlarkBootclasspath() {
    return Depset.of(Artifact.class, getBootclasspath().bootclasspath());
  }

  @Override
  public Depset getStarlarkJvmOptions() {
    return Depset.of(String.class, getJvmOptions());
  }

  @Override
  public Depset getStarlarkTools() {
    return Depset.of(Artifact.class, getTools());
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Nullable
  @Override
  public Artifact getTimezoneDataForStarlark(StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return getTimezoneData();
  }

  @Override
  public ImmutableList<String> getCompatibleJavacOptionsForStarlark(
      String key, StarlarkThread thread) throws EvalException {
    checkPrivateAccess(thread);
    return getCompatibleJavacOptions(key);
  }

  @AutoValue
  abstract static class JspecifyInfo {

    abstract JavaPluginData jspecifyProcessor();

    abstract NestedSet<Artifact> jspecifyImplicitDeps();

    abstract ImmutableList<String> jspecifyJavacopts();

    abstract ImmutableList<PackageSpecificationProvider> jspecifyPackages();

    boolean matches(Label label) {
      for (PackageSpecificationProvider provider : jspecifyPackages()) {
        for (PackageGroupContents specifications : provider.getPackageSpecifications().toList()) {
          if (specifications.containsPackage(label.getPackageIdentifier())) {
            return true;
          }
        }
      }
      return false;
    }

    static JspecifyInfo create(
        JavaPluginData jspecifyProcessor,
        NestedSet<Artifact> jspecifyImplicitDeps,
        ImmutableList<String> jspecifyJavacopts,
        ImmutableList<PackageSpecificationProvider> jspecifyPackages) {
      return new AutoValue_JavaToolchainProvider_JspecifyInfo(
          jspecifyProcessor, jspecifyImplicitDeps, jspecifyJavacopts, jspecifyPackages);
    }
  }
}
