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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction.LaunchInfo;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Substitution.ComputedSubstitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.cpp.CcLinkingInfo;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.rules.java.JavaCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Semantics for Bazel Java rules
 */
public class BazelJavaSemantics implements JavaSemantics {

  @AutoCodec public static final BazelJavaSemantics INSTANCE = new BazelJavaSemantics();

  private static final Template STUB_SCRIPT =
      Template.forResource(BazelJavaSemantics.class, "java_stub_template.txt");
  private static final String CLASSPATH_PLACEHOLDER = "%classpath%";
  private static final String RELATIVE_CLASSPATHS_PLACEHOLDER = "%relative_classpath%";
  private static final Template CLASSPATH_FILE_TEMPLATE =
      Template.forString(RELATIVE_CLASSPATHS_PLACEHOLDER);

  private static final String JAVABUILDER_CLASS_NAME =
      "com.google.devtools.build.buildjar.BazelJavaBuilder";

  private static final String JACOCO_COVERAGE_RUNNER_MAIN_CLASS =
      "com.google.testing.coverage.JacocoCoverageRunner";
  private static final String BAZEL_TEST_RUNNER_MAIN_CLASS =
      "com.google.testing.junit.runner.BazelTestRunner";
  private static final String EXPERIMENTAL_TEST_RUNNER_MAIN_CLASS =
      "com.google.testing.junit.runner.ExperimentalTestRunner";

  private BazelJavaSemantics() {
  }

  private boolean isJavaBinaryOrJavaTest(RuleContext ruleContext) {
    String ruleClass = ruleContext.getRule().getRuleClass();
    return ruleClass.equals("java_binary") || ruleClass.equals("java_test");
  }

  @Override
  public void checkRule(RuleContext ruleContext, JavaCommon javaCommon) {
  }

  @Override
  public void checkForProtoLibraryAndJavaProtoLibraryOnSameProto(
      RuleContext ruleContext, JavaCommon javaCommon) {}

  private static final String JUNIT4_RUNNER = "org.junit.runner.JUnitCore";

  @Nullable
  private String getMainClassInternal(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    if (!ruleContext.attributes().get("create_executable", Type.BOOLEAN)) {
      return null;
    }
    String mainClass = getMainClassFromRule(ruleContext);

    if (mainClass.isEmpty()) {
      mainClass = JavaCommon.determinePrimaryClass(ruleContext, sources);
    }
    return mainClass;
  }

  private String getMainClassFromRule(RuleContext ruleContext) {
    String mainClass = ruleContext.attributes().get("main_class", Type.STRING);
    if (!mainClass.isEmpty()) {
      return mainClass;
    }

    if (useLegacyJavaTest(ruleContext)) {
      // Legacy behavior for java_test rules: main_class defaulted to JUnit4 runner.
      // TODO(dmarting): remove once we drop the legacy bazel java_test behavior.
      if ("java_test".equals(ruleContext.getRule().getRuleClass())) {
        return JUNIT4_RUNNER;
      }
    } else {
      if (ruleContext.attributes().get("use_testrunner", Type.BOOLEAN)) {
        return ruleContext.getFragment(JavaConfiguration.class).useExperimentalTestRunner()
            ? EXPERIMENTAL_TEST_RUNNER_MAIN_CLASS
            : BAZEL_TEST_RUNNER_MAIN_CLASS;
      }
    }
    return mainClass;
  }

  private boolean isExperimentalJavaTest(RuleContext ruleContext) {
    return ruleContext.attributes().has("$experimental_testsupport")
        && TargetUtils.isTestRule(ruleContext.getRule())
        && getMainClassFromRule(ruleContext).equals(EXPERIMENTAL_TEST_RUNNER_MAIN_CLASS);
  }

  private void checkMainClass(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);
    String mainClass = getMainClassInternal(ruleContext, sources);

    if (!createExecutable && !isNullOrEmpty(mainClass)) {
      ruleContext.ruleError("main class must not be specified when executable is not created");
    }

    if (createExecutable && isNullOrEmpty(mainClass)) {
      if (sources.isEmpty()) {
        ruleContext.ruleError("need at least one of 'main_class' or Java source files");
      }
      mainClass = JavaCommon.determinePrimaryClass(ruleContext, sources);
      if (mainClass == null) {
        ruleContext.ruleError("cannot determine main class for launching "
                  + "(found neither a source file '" + ruleContext.getTarget().getName()
                  + ".java', nor a main_class attribute, and package name "
                  + "doesn't include 'java' or 'javatests')");
      }
    }
  }

  @Override
  public String getMainClass(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    checkMainClass(ruleContext, sources);
    return getMainClassInternal(ruleContext, sources);
  }

  @Override
  public ImmutableList<Artifact> collectResources(RuleContext ruleContext) {
    if (!ruleContext.getRule().isAttrDefined("resources", BuildType.LABEL_LIST)) {
      return ImmutableList.of();
    }

    return ruleContext.getPrerequisiteArtifacts("resources", Mode.TARGET).list();
  }

  /**
   * Used to generate the Classpaths contained within the stub.
   */
  private static class ComputedClasspathSubstitution extends ComputedSubstitution {
    private final NestedSet<Artifact> jars;
    private final String workspacePrefix;
    private final boolean isRunfilesEnabled;

    ComputedClasspathSubstitution(
        NestedSet<Artifact> jars, String workspacePrefix, boolean isRunfilesEnabled) {
      super(CLASSPATH_PLACEHOLDER);
      this.jars = jars;
      this.workspacePrefix = workspacePrefix;
      this.isRunfilesEnabled = isRunfilesEnabled;
    }

    /**
     * Builds a class path by concatenating the root relative paths of the artifacts. Each relative
     * path entry is prepended with "${RUNPATH}" which will be expanded by the stub script at
     * runtime, to either "${JAVA_RUNFILES}/" or if we are lucky, the empty string.
     */
    @Override
    public String getValue() {
      StringBuilder buffer = new StringBuilder();
      buffer.append("\"");
      for (Artifact artifact : jars) {
        if (buffer.length() > 1) {
          buffer.append(File.pathSeparatorChar);
        }
        if (!isRunfilesEnabled) {
          buffer.append("$(rlocation ");
          PathFragment runfilePath =
              PathFragment.create(workspacePrefix).getRelative(artifact.getRunfilesPath());
          buffer.append(runfilePath.getPathString());
          buffer.append(")");
        } else {
          buffer.append("${RUNPATH}");
          buffer.append(artifact.getRunfilesPath().getPathString());
        }
      }
      buffer.append("\"");
      return buffer.toString();
    }
  }

  /**
   * In Bazel this {@code createStubAction} considers {@code javaExecutable} as a file path for the
   * JVM binary (java).
   */
  @Override
  public boolean isJavaExecutableSubstitution() {
    return false;
  }

  @Override
  public Artifact createStubAction(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact executable,
      String javaStartClass,
      String javaExecutable) {
    return createStubAction(
        ruleContext, javaCommon, jvmFlags, executable, javaStartClass, "",
        NestedSetBuilder.<Artifact>stableOrder(), javaExecutable);
  }

  @Override
  public Artifact createStubAction(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact executable,
      String javaStartClass,
      String coverageStartClass,
      NestedSetBuilder<Artifact> filesBuilder,
      String javaExecutable) {
    Preconditions.checkState(ruleContext.getConfiguration().hasFragment(JavaConfiguration.class));

    Preconditions.checkNotNull(jvmFlags);
    Preconditions.checkNotNull(executable);
    Preconditions.checkNotNull(javaStartClass);
    Preconditions.checkNotNull(javaExecutable);

    List<Substitution> arguments = new ArrayList<>();
    String workspaceName = ruleContext.getWorkspaceName();
    final String workspacePrefix = workspaceName + (workspaceName.isEmpty() ? "" : "/");
    final boolean isRunfilesEnabled = ruleContext.getConfiguration().runfilesEnabled();
    arguments.add(Substitution.of("%runfiles_manifest_only%", isRunfilesEnabled ? "" : "1"));
    arguments.add(Substitution.of("%workspace_prefix%", workspacePrefix));
    arguments.add(
        Substitution.of(
            "%javabin%",
            JavaCommon.getJavaBinSubstitutionFromJavaExecutable(ruleContext, javaExecutable)));
    arguments.add(Substitution.of("%needs_runfiles%",
        JavaCommon.getJavaExecutable(ruleContext).isAbsolute() ? "0" : "1"));

    TransitiveInfoCollection testSupport = getTestSupport(ruleContext);
    NestedSet<Artifact> testSupportJars =
        testSupport == null
            ? NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER)
            : getRuntimeJarsForTargets(testSupport);

    NestedSetBuilder<Artifact> classpathBuilder = NestedSetBuilder.naiveLinkOrder();
    classpathBuilder.addTransitive(javaCommon.getRuntimeClasspath());
    if (enforceExplicitJavaTestDeps(ruleContext)) {
      // Currently, this is only needed when --explicit_java_test_deps=true, as otherwise the
      // testSupport classpath is wrongly present in the javaCommon.getRuntimeClasspath().
      classpathBuilder.addTransitive(testSupportJars);
    }
    NestedSet<Artifact> classpath = classpathBuilder.build();

    if (isExperimentalJavaTest(ruleContext)) {
      if (!isRunfilesEnabled) {
        ruleContext.ruleError(
            "ExperimentalTestRunner can't work on Windows since Windows doesn't support runfiles.");
      }
      if (testSupport == null) {
        // This may happen when the user sets use_testrunner=0 and manually chooses
        // main_class=ExperimentalTestRunner.
        ruleContext.ruleError("Unexpected usage of ExperimentalTestRunner.");
      }
      // Keep only the locations containing the classes to start the test runner itself within,
      // classpath variable, and place all the paths required for the test run in a classpaths file,
      // so that the classes for the test target may be loaded by a separate ClassLoader.
      classpath = testSupportJars;
    }
    arguments.add(new ComputedClasspathSubstitution(classpath, workspacePrefix, isRunfilesEnabled));

    JavaCompilationArtifacts javaArtifacts = javaCommon.getJavaCompilationArtifacts();
    String path =
        javaArtifacts.getInstrumentedJar() != null
            ? "${JAVA_RUNFILES}/"
            + workspacePrefix
            + javaArtifacts.getInstrumentedJar().getRootRelativePath().getPathString()
            : "";

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()
        && ruleContext.getConfiguration().isExperimentalJavaCoverage()) {
      Artifact runtimeClassPathArtifact = ruleContext.getUniqueDirectoryArtifact(
          "coverage_runtime_classpath",
          "runtime-classpath.txt",
          ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(new LazyWritePathsFileAction(
          ruleContext.getActionOwner(),
          runtimeClassPathArtifact,
          javaCommon.getRuntimeClasspath(),
          true));
      filesBuilder.add(runtimeClassPathArtifact);
      arguments.add(Substitution.of(
          JavaSemantics.JACOCO_METADATA_PLACEHOLDER,
          "export JACOCO_METADATA_JAR=${JAVA_RUNFILES}/" + workspacePrefix + "/"
              + runtimeClassPathArtifact.getRootRelativePathString()
      ));
      arguments.add(Substitution.of(
          JavaSemantics.JACOCO_MAIN_CLASS_PLACEHOLDER,
          "export JACOCO_MAIN_CLASS=" + coverageStartClass));
      arguments.add(Substitution.of(
          JavaSemantics.JACOCO_JAVA_RUNFILES_ROOT_PLACEHOLDER,
          "export JACOCO_JAVA_RUNFILES_ROOT=${JAVA_RUNFILES}/" + workspacePrefix)
      );
      arguments.add(
          Substitution.of("%java_start_class%", ShellEscaper.escapeString(javaStartClass)));
    } else {
      arguments.add(Substitution.of(JavaSemantics.JACOCO_METADATA_PLACEHOLDER,
          ruleContext.getConfiguration().isCodeCoverageEnabled()
              ? "export JACOCO_METADATA_JAR=" + path : ""));
      arguments.add(Substitution.of(JavaSemantics.JACOCO_MAIN_CLASS_PLACEHOLDER, ""));
      arguments.add(Substitution.of(JavaSemantics.JACOCO_JAVA_RUNFILES_ROOT_PLACEHOLDER, ""));
    }

    arguments.add(Substitution.of("%java_start_class%",
        ShellEscaper.escapeString(javaStartClass)));

    ImmutableList<String> jvmFlagsList = ImmutableList.copyOf(jvmFlags);
    arguments.add(Substitution.ofSpaceSeparatedList("%jvm_flags%", jvmFlagsList));

    if (OS.getCurrent() == OS.WINDOWS) {
      return createWindowsExeLauncher(
          ruleContext,
          javaExecutable,
          classpath,
          javaStartClass,
          jvmFlagsList,
          executable);
    }

    ruleContext.registerAction(new TemplateExpansionAction(
        ruleContext.getActionOwner(), executable, STUB_SCRIPT, arguments, true));
    return executable;
  }

  private static Artifact createWindowsExeLauncher(
      RuleContext ruleContext,
      String javaExecutable,
      NestedSet<Artifact> classpath,
      String javaStartClass,
      ImmutableList<String> jvmFlags,
      Artifact javaLauncher) {

    LaunchInfo launchInfo =
        LaunchInfo.builder()
            .addKeyValuePair("binary_type", "Java")
            .addKeyValuePair("workspace_name", ruleContext.getWorkspaceName())
            .addKeyValuePair(
                "symlink_runfiles_enabled",
                ruleContext.getConfiguration().runfilesEnabled() ? "1" : "0")
            .addKeyValuePair("java_bin_path", javaExecutable)
            .addKeyValuePair(
                "jar_bin_path",
                JavaCommon.getJavaExecutable(ruleContext)
                    .getParentDirectory()
                    .getRelative("jar.exe")
                    .getPathString())
            .addKeyValuePair("java_start_class", javaStartClass)
            .addJoinedValues(
                "classpath",
                ";",
                Iterables.transform(classpath, Artifact.ROOT_RELATIVE_PATH_STRING))
            .addJoinedValues("jvm_flags", " ", jvmFlags)
            .build();

    LauncherFileWriteAction.createAndRegister(ruleContext, javaLauncher, launchInfo);

    return javaLauncher;
  }

  private static boolean enforceExplicitJavaTestDeps(RuleContext ruleContext) {
    return ruleContext.getFragment(JavaConfiguration.class).explicitJavaTestDeps();
  }

  /**
   * Substitutes the placeholder with {@link File#pathSeparatorChar} separated relative classpaths.
   */
  private static class ComputedRelativeClasspathsSubstitution extends ComputedSubstitution {
    private final JavaCommon javaCommon;
    public ComputedRelativeClasspathsSubstitution(JavaCommon javaCommon) {
      super(RELATIVE_CLASSPATHS_PLACEHOLDER);
      this.javaCommon = javaCommon;
    }

    @Override
    public String getValue() {
      // TODO(kush): Get this to work when runfilesEnabled=false, like in windows.
      Iterable<String> paths =
          Iterables.transform(
              javaCommon.getRuntimeClasspath(),
              artifact -> artifact == null ? null : artifact.getRunfilesPathString());
      return Joiner.on(File.pathSeparatorChar).skipNulls().join(paths);
    }
  }

  @Override
  public Optional<Artifact> createClasspathsFile(RuleContext ruleContext, JavaCommon javaCommon)
      throws InterruptedException {
    if (!isExperimentalJavaTest(ruleContext)) {
      return Optional.absent();
    }
    Artifact classpathFile = ruleContext.getImplicitOutputArtifact(JAVA_TEST_CLASSPATHS_FILE);
    List<Substitution> substitutions =
        ImmutableList.<Substitution>of(new ComputedRelativeClasspathsSubstitution(javaCommon));

    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            classpathFile,
            CLASSPATH_FILE_TEMPLATE,
            substitutions,
            /*makeExecutable=*/ false));
    return Optional.of(classpathFile);
  }

  @Nullable
  private TransitiveInfoCollection getTestSupport(RuleContext ruleContext) {
    if (!isJavaBinaryOrJavaTest(ruleContext)) {
      return null;
    }
    if (useLegacyJavaTest(ruleContext)) {
      return null;
    }

    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);
    if (createExecutable && ruleContext.attributes().get("use_testrunner", Type.BOOLEAN)) {
      String testSupport =
          isExperimentalJavaTest(ruleContext) ? "$experimental_testsupport" : "$testsupport";
      return Iterables.getOnlyElement(ruleContext.getPrerequisites(testSupport, Mode.TARGET));
    } else {
      return null;
    }
  }

  private static NestedSet<Artifact> getRuntimeJarsForTargets(TransitiveInfoCollection... deps) {
    // The dep may be a simple JAR and not a java rule, hence we can't simply do
    // dep.getProvider(JavaCompilationArgsProvider.class).getRecursiveJavaCompilationArgs(),
    // so we reuse the logic within JavaCompilationArgsProvider to handle both scenarios.
    return JavaCompilationArgsProvider.legacyFromTargets(ImmutableList.copyOf(deps))
        .getRuntimeJars();
  }

  @Override
  public void addRunfilesForBinary(RuleContext ruleContext, Artifact launcher,
      Runfiles.Builder runfilesBuilder) {
    TransitiveInfoCollection testSupport = getTestSupport(ruleContext);
    if (testSupport != null) {
      // We assume that the runtime jars will not have conflicting artifacts
      // with the same root relative path
      runfilesBuilder.addTransitiveArtifactsWrappedInStableOrder(
          getRuntimeJarsForTargets(testSupport));
    }
  }

  @Override
  public void addRunfilesForLibrary(RuleContext ruleContext, Runfiles.Builder runfilesBuilder) {
  }

  @Override
  public void collectTargetsTreatedAsDeps(
      RuleContext ruleContext,
      ImmutableList.Builder<TransitiveInfoCollection> builder,
      ClasspathType type) {
    if (type == ClasspathType.COMPILE_ONLY && enforceExplicitJavaTestDeps(ruleContext)) {
      // We add the test support below, but the test framework's deps are not relevant for
      // COMPILE_ONLY, hence we return here.
      // TODO(bazel-team): Ideally enforceExplicitJavaTestDeps should be the default behaviour,
      // since the testSupport deps don't belong to the COMPILE_ONLY classpath, but since many
      // targets may break, we are keeping it behind this flag.
      return;
    }
    TransitiveInfoCollection testSupport = getTestSupport(ruleContext);
    if (testSupport != null) {
      builder.add(testSupport);
    }
  }

  @Override
  public ImmutableList<String> getCompatibleJavacOptions(
      RuleContext ruleContext, JavaToolchainProvider toolchain) {
    return ImmutableList.of();
  }

  @Override
  public void addProviders(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      Artifact gensrcJar,
      RuleConfiguredTargetBuilder ruleBuilder) {
    // TODO(plf): Figure out whether we can remove support for C++ dependencies in Bazel.
    ImmutableList<? extends TransitiveInfoCollection> deps =
        javaCommon.targetsTreatedAsDeps(ClasspathType.BOTH);
    ImmutableList<CcLinkingInfo> ccLinkingInfos =
        ImmutableList.<CcLinkingInfo>builder()
            .addAll(AnalysisUtils.getProviders(deps, CcLinkingInfo.PROVIDER))
            .addAll(
                Streams.stream(AnalysisUtils.getProviders(deps, JavaCcLinkParamsProvider.PROVIDER))
                    .map(JavaCcLinkParamsProvider::getCcLinkingInfo)
                    .collect(ImmutableList.toImmutableList()))
            .build();

    // TODO(plf): return empty CcLinkingInfo because deps= in Java targets should not contain C++
    // targets. We need to make sure that no one uses this functionality, though.
    ruleBuilder.addNativeDeclaredProvider(CcLinkingInfo.merge(ccLinkingInfos));
  }

  // TODO(dmarting): simplify that logic when we remove the legacy Bazel java_test behavior.
  private String getPrimaryClassLegacy(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);
    if (!createExecutable) {
      return null;
    }
    return getMainClassInternal(ruleContext, sources);
  }

  private String getPrimaryClassNew(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);

    if (!createExecutable) {
      return null;
    }

    boolean useTestrunner = ruleContext.attributes().get("use_testrunner", Type.BOOLEAN);

    String testClass = ruleContext.getRule().isAttrDefined("test_class", Type.STRING)
        ? ruleContext.attributes().get("test_class", Type.STRING) : "";

    if (useTestrunner) {
      if (testClass.isEmpty()) {
        testClass = JavaCommon.determinePrimaryClass(ruleContext, sources);
        if (testClass == null) {
          ruleContext.ruleError("cannot determine junit.framework.Test class "
                    + "(Found no source file '" + ruleContext.getTarget().getName()
                    + ".java' and package name doesn't include 'java' or 'javatests'. "
                    + "You might want to rename the rule or add a 'test_class' "
                    + "attribute.)");
        }
      }
      return testClass;
    } else {
      if (!testClass.isEmpty()) {
        ruleContext.attributeError("test_class", "this attribute is only meaningful to "
            + "BazelTestRunner, but you are not using it (use_testrunner = 0)");
      }
      return getMainClassInternal(ruleContext, sources);
    }
  }

  @Override
  public String getPrimaryClass(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    return useLegacyJavaTest(ruleContext)
        ? getPrimaryClassLegacy(ruleContext, sources)
        : getPrimaryClassNew(ruleContext, sources);
  }

  @Override
  public Iterable<String> getJvmFlags(
      RuleContext ruleContext, ImmutableList<Artifact> sources, List<String> userJvmFlags) {
    ImmutableList.Builder<String> jvmFlags = ImmutableList.builder();
    jvmFlags.addAll(userJvmFlags);

    if (!useLegacyJavaTest(ruleContext)) {
      if (ruleContext.attributes().get("use_testrunner", Type.BOOLEAN)) {
        String testClass = ruleContext.getRule().isAttrDefined("test_class", Type.STRING)
            ? ruleContext.attributes().get("test_class", Type.STRING) : "";
        if (testClass.isEmpty()) {
          testClass = JavaCommon.determinePrimaryClass(ruleContext, sources);
        }

        if (testClass == null) {
          ruleContext.ruleError("cannot determine test class");
        } else {
          // Always run junit tests with -ea (enable assertion)
          jvmFlags.add("-ea");
          // "suite" is a misnomer.
          jvmFlags.add("-Dbazel.test_suite=" +  ShellEscaper.escapeString(testClass));
        }
      }
    }

    return jvmFlags.build();
  }

  /**
   * Returns whether coverage has instrumented artifacts.
   */
  public static boolean hasInstrumentationMetadata(JavaTargetAttributes.Builder attributes) {
    return !attributes.getInstrumentationMetadata().isEmpty();
  }

  @Override
  public String addCoverageSupport(
      JavaCompilationHelper helper,
      JavaTargetAttributes.Builder attributes,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      String mainClass) throws InterruptedException {
    return addCoverageSupport(helper, attributes, executable, instrumentationMetadata,
        javaArtifactsBuilder, mainClass, false);
  }

  // TODO(yueg): refactor this (only mainClass different for now)
  @Override
  public String addCoverageSupport(
      JavaCompilationHelper helper,
      JavaTargetAttributes.Builder attributes,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      String mainClass,
      boolean isExperimentalCoverage)
      throws InterruptedException {
    // This method can be called only for *_binary/*_test targets.
    Preconditions.checkNotNull(executable);
    if (!isExperimentalCoverage) {
      // Add our own metadata artifact (if any).
      if (instrumentationMetadata != null) {
        attributes.addInstrumentationMetadataEntries(ImmutableList.of(instrumentationMetadata));
      }

      if (!hasInstrumentationMetadata(attributes)) {
        return mainClass;
      }

      Artifact instrumentedJar =
          helper
              .getRuleContext()
              .getBinArtifact(helper.getRuleContext().getLabel().getName() + "_instrumented.jar");

      // Create an instrumented Jar. This will be referenced on the runtime classpath prior
      // to all other Jars.
      JavaCommon.createInstrumentedJarAction(
          helper.getRuleContext(),
          this,
          attributes.getInstrumentationMetadata(),
          instrumentedJar,
          mainClass);
      javaArtifactsBuilder.setInstrumentedJar(instrumentedJar);
    }

    // Add the coverage runner to the list of dependencies when compiling in coverage mode.
    TransitiveInfoCollection runnerTarget =
        helper.getRuleContext().getPrerequisite("$jacocorunner", Mode.TARGET);
    if (JavaInfo.getProvider(JavaCompilationArgsProvider.class, runnerTarget) != null) {
      helper.addLibrariesToAttributes(ImmutableList.of(runnerTarget));
    } else {
      helper
          .getRuleContext()
          .ruleError(
              "this rule depends on "
                  + helper.getRuleContext().attributes().get("$jacocorunner", BuildType.LABEL)
                  + " which is not a java_library rule, or contains errors");
    }

    // We do not add the instrumented jar to the runtime classpath, but provide it in the shell
    // script via an environment variable.
    return JACOCO_COVERAGE_RUNNER_MAIN_CLASS;
  }

  @Override
  public CustomCommandLine buildSingleJarCommandLine(
      String toolchainIdentifier,
      Artifact output,
      String mainClass,
      ImmutableList<String> manifestLines,
      Iterable<Artifact> buildInfoFiles,
      ImmutableList<Artifact> resources,
      NestedSet<Artifact> classpath,
      boolean includeBuildData,
      Compression compression,
      Artifact launcher,
      boolean usingNativeSinglejar,
      // Explicitly ignoring params since Bazel doesn't yet support one version
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      Artifact oneVersionWhitelistArtifact) {
    return DeployArchiveBuilder.defaultSingleJarCommandLineWithoutOneVersion(
            output,
            mainClass,
            manifestLines,
            buildInfoFiles,
            resources,
            classpath,
            includeBuildData,
            compression,
            launcher,
            usingNativeSinglejar)
        .build();
  }

  @Override
  public ImmutableList<Artifact> translate(RuleContext ruleContext, JavaConfiguration javaConfig,
      List<Artifact> messages) {
    return ImmutableList.<Artifact>of();
  }

  @Override
  public Pair<Artifact, Artifact> getLauncher(
      RuleContext ruleContext,
      JavaCommon common,
      DeployArchiveBuilder deployArchiveBuilder,
      DeployArchiveBuilder unstrippedDeployArchiveBuilder,
      Runfiles.Builder runfilesBuilder,
      List<String> jvmFlags,
      JavaTargetAttributes.Builder attributesBuilder,
      boolean shouldStrip) {
    Artifact launcher = JavaHelper.launcherArtifactForTarget(this, ruleContext);
    return new Pair<>(launcher, launcher);
  }

  @Override
  public void addArtifactToJavaTargetAttribute(JavaTargetAttributes.Builder builder,
      Artifact srcArtifact) {
  }

  @Override
  public PathFragment getDefaultJavaResourcePath(PathFragment path) {
    // Look for src/.../resources to match Maven repository structure.
    List<String> segments = path.getSegments();
    for (int i = 0; i < segments.size() - 2; ++i) {
      if (segments.get(i).equals("src") && segments.get(i + 2).equals("resources")) {
        return path.subFragment(i + 3);
      }
    }
    PathFragment javaPath = JavaUtil.getJavaPath(path);
    return javaPath == null ? path : javaPath;
  }

  @Override
  public List<String> getExtraArguments(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    if (ruleContext.getRule().getRuleClass().equals("java_test")) {
      if (useLegacyJavaTest(ruleContext)) {
        TestConfiguration testConfiguration =
            ruleContext.getConfiguration().getFragment(TestConfiguration.class);
        if (testConfiguration.getTestArguments().isEmpty()
            && !ruleContext.attributes().isAttributeValueExplicitlySpecified("args")) {
          ImmutableList.Builder<String> builder = ImmutableList.builder();
          for (Artifact artifact : sources) {
            PathFragment path = artifact.getRootRelativePath();
            String className = JavaUtil.getJavaFullClassname(FileSystemUtils.removeExtension(path));
            if (className != null) {
              builder.add(className);
            }
          }
          return builder.build();
        }
      }
    }
    return ImmutableList.<String>of();
  }

  private boolean useLegacyJavaTest(RuleContext ruleContext) {
    return !ruleContext.attributes().isAttributeValueExplicitlySpecified("test_class")
        && ruleContext.getFragment(JavaConfiguration.class).useLegacyBazelJavaTest();
  }

  @Override
  public String getJavaBuilderMainClass() {
    return JAVABUILDER_CLASS_NAME;
  }

  @Override
  public Artifact getProtoMapping(RuleContext ruleContext) throws InterruptedException {
    return null;
  }

  @Nullable
  @Override
  public GeneratedExtensionRegistryProvider createGeneratedExtensionRegistry(
      RuleContext ruleContext,
      JavaCommon common,
      NestedSetBuilder<Artifact> filesBuilder,
      JavaCompilationArtifacts.Builder javaCompilationArtifactsBuilder,
      JavaRuleOutputJarsProvider.Builder javaRuleOutputJarsProviderBuilder,
      JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder)
    throws InterruptedException {
    return null;
  }

  @Override
  public Artifact getObfuscatedConstantStringMap(RuleContext ruleContext)
      throws InterruptedException {
    return null;
  }

  @Override
  public boolean isJavaProtoLibraryStrictDeps(RuleContext ruleContext) {
    return false;
  }
}
