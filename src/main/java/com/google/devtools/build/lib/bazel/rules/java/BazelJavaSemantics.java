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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.Runfiles.Builder;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction.LaunchInfo;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Substitution.ComputedSubstitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
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
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
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

  private static final String JAVABUILDER_CLASS_NAME =
      "com.google.devtools.build.buildjar.BazelJavaBuilder";

  private static final String JACOCO_COVERAGE_RUNNER_MAIN_CLASS =
      "com.google.testing.coverage.JacocoCoverageRunner";
  private static final String BAZEL_TEST_RUNNER_MAIN_CLASS =
      "com.google.testing.junit.runner.BazelTestRunner";

  private BazelJavaSemantics() {
  }

  @Override
  public void checkRule(RuleContext ruleContext, JavaCommon javaCommon) {
  }

  @Override
  public void checkForProtoLibraryAndJavaProtoLibraryOnSameProto(
      RuleContext ruleContext, JavaCommon javaCommon) {}

  private static final String JUNIT4_RUNNER = "org.junit.runner.JUnitCore";

  @Override
  public String getTestRunnerMainClass() {
    return BAZEL_TEST_RUNNER_MAIN_CLASS;
  }

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

    if (JavaSemantics.useLegacyJavaTest(ruleContext)) {
      // Legacy behavior for java_test rules: main_class defaulted to JUnit4 runner.
      // TODO(dmarting): remove once we drop the legacy bazel java_test behavior.
      if ("java_test".equals(ruleContext.getRule().getRuleClass())) {
        return JUNIT4_RUNNER;
      }
    } else {
      if (ruleContext.attributes().get("use_testrunner", Type.BOOLEAN)) {
        return BAZEL_TEST_RUNNER_MAIN_CLASS;
      }
    }
    return mainClass;
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
        ruleContext.ruleError(
            String.format(
                "main_class was not provided and cannot be inferred: source path doesn't include"
                    + " a known root (%s)",
                Joiner.on(", ").join(JavaUtil.KNOWN_SOURCE_ROOTS)));
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

    return ruleContext.getPrerequisiteArtifacts("resources", TransitionMode.TARGET).list();
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
      for (Artifact artifact : jars.toList()) {
        if (buffer.length() > 1) {
          buffer.append(File.pathSeparatorChar);
        }
        if (!isRunfilesEnabled) {
          buffer.append("$(rlocation ");
          PathFragment runfilePath =
              PathFragment.create(workspacePrefix).getRelative(artifact.getRootRelativePath());
          buffer.append(runfilePath.getPathString());
          buffer.append(")");
        } else {
          buffer.append("${RUNPATH}");
          buffer.append(artifact.getRootRelativePath().getPathString());
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
        ruleContext,
        javaCommon,
        jvmFlags,
        executable,
        javaStartClass,
        "",
        NestedSetBuilder.<Artifact>stableOrder(),
        javaExecutable,
        /* createCoverageMetadataJar= */ true);
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
      String javaExecutable,
      boolean createCoverageMetadataJar) {
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

    TransitiveInfoCollection testSupport = JavaSemantics.getTestSupport(ruleContext);
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

    if (JavaSemantics.isTestTargetAndPersistentTestRunner(ruleContext)) {
      // Create an artifact that stores the test's runtime classpath (excluding the test support
      // classpath). The file is read by the test runner. The jars inside the file are loaded
      // dynamically for every test run into a custom classloader.
      arguments.add(
          new ComputedClasspathSubstitution(
              JavaSemantics.getTestSupportRuntimeClasspath(ruleContext),
              workspacePrefix,
              isRunfilesEnabled));

      // Create an artifact that stores the test's runtime classpath.
      Artifact testRuntimeClasspathArtifact =
          ruleContext.getBinArtifact(
              ruleContext.getLabel().getName() + "_test_runtime_classpath.txt");
      ruleContext.registerAction(
          new LazyWritePathsFileAction(
              ruleContext.getActionOwner(),
              testRuntimeClasspathArtifact,
              javaCommon.getRuntimeClasspath(),
              /* filesToIgnore= */ ImmutableSet.of(),
              true));
      filesBuilder.add(testRuntimeClasspathArtifact);

      arguments.add(
          Substitution.of(
              TEST_RUNTIME_CLASSPATH_FILE_PLACEHOLDER,
              "export WORKSPACE_PREFIX="
                  + workspacePrefix
                  + "\nexport TEST_RUNTIME_CLASSPATH_FILE=${JAVA_RUNFILES}"
                  + File.separator
                  + workspacePrefix
                  + testRuntimeClasspathArtifact.getRootRelativePathString()));
    } else {
      arguments.add(
          new ComputedClasspathSubstitution(classpath, workspacePrefix, isRunfilesEnabled));
      arguments.add(Substitution.of(TEST_RUNTIME_CLASSPATH_FILE_PLACEHOLDER, ""));
    }

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      if (createCoverageMetadataJar) {
        Artifact runtimeClassPathArtifact =
            ruleContext.getUniqueDirectoryArtifact(
                "coverage_runtime_classpath",
                "runtime-classpath.txt",
                ruleContext.getBinOrGenfilesDirectory());
        ruleContext.registerAction(
            new LazyWritePathsFileAction(
                ruleContext.getActionOwner(),
                runtimeClassPathArtifact,
                javaCommon.getRuntimeClasspath(),
                /* filesToIgnore= */ ImmutableSet.of(),
                true));
        filesBuilder.add(runtimeClassPathArtifact);
        arguments.add(
            Substitution.of(
                JavaSemantics.JACOCO_METADATA_PLACEHOLDER,
                "export JACOCO_METADATA_JAR=${JAVA_RUNFILES}/"
                    + workspacePrefix
                    + "/"
                    + runtimeClassPathArtifact.getRootRelativePathString()));
      } else {
        // Remove the placeholder in the stub otherwise bazel coverage fails.
        arguments.add(Substitution.of(JavaSemantics.JACOCO_METADATA_PLACEHOLDER, ""));
      }
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
      arguments.add(Substitution.of(JavaSemantics.JACOCO_METADATA_PLACEHOLDER, ""));
      arguments.add(Substitution.of(JavaSemantics.JACOCO_MAIN_CLASS_PLACEHOLDER, ""));
      arguments.add(Substitution.of(JavaSemantics.JACOCO_JAVA_RUNFILES_ROOT_PLACEHOLDER, ""));
    }

    arguments.add(Substitution.of("%java_start_class%",
        ShellEscaper.escapeString(javaStartClass)));

    ImmutableList<String> jvmFlagsList = ImmutableList.copyOf(jvmFlags);
    arguments.add(Substitution.ofSpaceSeparatedList("%jvm_flags%", jvmFlagsList));

    if (OS.getCurrent() == OS.WINDOWS) {
      List<String> jvmFlagsForLauncher = jvmFlagsList;
      try {
        jvmFlagsForLauncher = new ArrayList<>(jvmFlagsList.size());
        for (String f : jvmFlagsList) {
          ShellUtils.tokenize(jvmFlagsForLauncher, f);
        }
      } catch (TokenizationException e) {
        ruleContext.attributeError("jvm_flags", "could not Bash-tokenize flag: " + e);
      }

      return createWindowsExeLauncher(
          ruleContext, javaExecutable, classpath, javaStartClass, jvmFlagsForLauncher, executable);
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
      List<String> jvmFlags,
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
                Iterables.transform(classpath.toList(), Artifact.ROOT_RELATIVE_PATH_STRING))
            // TODO(laszlocsomor): Change the Launcher to accept multiple jvm_flags entries. As of
            // 2019-02-13 the Launcher accepts just one jvm_flags entry, which contains all the
            // flags, joined by TAB characters. The Launcher splits up the string to get the
            // individual jvm_flags. This approach breaks with flags that contain a TAB character.
            .addJoinedValues("jvm_flags", "\t", jvmFlags)
            .build();

    LauncherFileWriteAction.createAndRegister(ruleContext, javaLauncher, launchInfo);

    return javaLauncher;
  }

  private static boolean enforceExplicitJavaTestDeps(RuleContext ruleContext) {
    return ruleContext.getFragment(JavaConfiguration.class).explicitJavaTestDeps();
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
    TransitiveInfoCollection testSupport = JavaSemantics.getTestSupport(ruleContext);
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
    if (!JavaSemantics.isTestTargetAndPersistentTestRunner(ruleContext)) {
      // Only add the test support to the dependencies when running in regular mode.
      // In persistent test runner mode don't pollute the classpath of the test with
      // the test support classes.
      TransitiveInfoCollection testSupport = JavaSemantics.getTestSupport(ruleContext);
      if (testSupport != null) {
        builder.add(testSupport);
      }
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
    ImmutableList<CcInfo> ccInfos =
        ImmutableList.<CcInfo>builder()
            .addAll(AnalysisUtils.getProviders(deps, CcInfo.PROVIDER))
            .addAll(
                Streams.stream(AnalysisUtils.getProviders(deps, JavaCcLinkParamsProvider.PROVIDER))
                    .map(JavaCcLinkParamsProvider::getCcInfo)
                    .collect(ImmutableList.toImmutableList()))
            .build();

    // TODO(plf): return empty CcLinkingInfo because deps= in Java targets should not contain C++
    // targets. We need to make sure that no one uses this functionality, though.
    ruleBuilder.addNativeDeclaredProvider(new JavaCcLinkParamsProvider(CcInfo.merge(ccInfos)));
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
    return JavaSemantics.useLegacyJavaTest(ruleContext)
        ? getPrimaryClassLegacy(ruleContext, sources)
        : getPrimaryClassNew(ruleContext, sources);
  }

  @Override
  public Iterable<String> getJvmFlags(
      RuleContext ruleContext, ImmutableList<Artifact> sources, List<String> userJvmFlags) {
    ImmutableList.Builder<String> jvmFlags = ImmutableList.builder();
    jvmFlags.addAll(userJvmFlags);

    if (!JavaSemantics.useLegacyJavaTest(ruleContext)) {
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

  @Override
  public String addCoverageSupport(JavaCompilationHelper helper, Artifact executable) {
    // This method can be called only for *_binary/*_test targets.
    Preconditions.checkNotNull(executable);
    if (!helper.addCoverageSupport()) {
      // Fallback to $jacocorunner attribute if no jacocorunner was found in the toolchain.

      // Add the coverage runner to the list of dependencies when compiling in coverage mode.
      TransitiveInfoCollection runnerTarget =
          helper.getRuleContext().getPrerequisite("$jacocorunner", TransitionMode.TARGET);
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
      Artifact oneVersionWhitelistArtifact,
      Artifact sharedArchive) {
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
      Builder runfilesBuilder,
      List<String> jvmFlags,
      JavaTargetAttributes.Builder attributesBuilder,
      boolean shouldStrip,
      CcToolchainProvider ccToolchain,
      FeatureConfiguration featureConfiguration) {
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
      if (JavaSemantics.useLegacyJavaTest(ruleContext)) {
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
  public void checkDependencyRuleKinds(RuleContext ruleContext) {}
}

