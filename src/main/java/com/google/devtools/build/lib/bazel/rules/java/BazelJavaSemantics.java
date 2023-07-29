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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.LauncherFileWriteAction.LaunchInfo;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Substitution.ComputedSubstitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.rules.java.JavaBuildInfoFactory;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Semantics for Bazel Java rules
 */
public class BazelJavaSemantics implements JavaSemantics {

  @SerializationConstant public static final BazelJavaSemantics INSTANCE = new BazelJavaSemantics();

  private static final Template STUB_SCRIPT =
      Template.forResource(BazelJavaSemantics.class, "java_stub_template.txt");
  private static final String CLASSPATH_PLACEHOLDER = "%classpath%";

  private static final String JACOCO_COVERAGE_RUNNER_MAIN_CLASS =
      "com.google.testing.coverage.JacocoCoverageRunner";
  private static final String BAZEL_TEST_RUNNER_MAIN_CLASS =
      "com.google.testing.junit.runner.BazelTestRunner";

  private BazelJavaSemantics() {
  }

  private static final String JAVA_TOOLCHAIN_TYPE = "@bazel_tools//tools/jdk:toolchain_type";

  @Override
  public String getJavaToolchainType() {
    return JAVA_TOOLCHAIN_TYPE;
  }

  @Override
  public void checkRule(RuleContext ruleContext, JavaCommon javaCommon) {
  }

  @Override
  public ImmutableList<Artifact> getBuildInfo(RuleContext ruleContext, int stamp)
      throws RuleErrorException, InterruptedException {
    return ruleContext.getBuildInfo(JavaBuildInfoFactory.KEY);
  }

  @Override
  public void checkForProtoLibraryAndJavaProtoLibraryOnSameProto(
      RuleContext ruleContext, JavaCommon javaCommon) {}

  @Override
  public String getTestRunnerMainClass() {
    return BAZEL_TEST_RUNNER_MAIN_CLASS;
  }

  @Override
  public ImmutableList<Artifact> collectResources(RuleContext ruleContext) {
    if (!ruleContext.getRule().isAttrDefined("resources", BuildType.LABEL_LIST)) {
      return ImmutableList.of();
    }

    return ruleContext.getPrerequisiteArtifacts("resources").list();
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
      String coverageStartClass,
      NestedSetBuilder<Artifact> filesBuilder,
      String javaExecutable,
      boolean createCoverageMetadataJar)
      throws RuleErrorException {
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

    arguments.add(new ComputedClasspathSubstitution(classpath, workspacePrefix, isRunfilesEnabled));

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
                    + runtimeClassPathArtifact.getRepositoryRelativePathString()));
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
                Iterables.transform(classpath.toList(), Artifact.RUNFILES_PATH_STRING))
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

  private static NestedSet<Artifact> getRuntimeJarsForTargets(TransitiveInfoCollection... deps)
      throws RuleErrorException {
    // The dep may be a simple JAR and not a java rule, hence we can't simply do
    // dep.getProvider(JavaCompilationArgsProvider.class).getRecursiveJavaCompilationArgs(),
    // so we reuse the logic within JavaCompilationArgsProvider to handle both scenarios.
    return JavaCompilationArgsProvider.legacyFromTargets(ImmutableList.copyOf(deps))
        .getRuntimeJars();
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
    // Only add the test support to the dependencies when running in regular mode.
    // In persistent test runner mode don't pollute the classpath of the test with
    // the test support classes.
    TransitiveInfoCollection testSupport = JavaSemantics.getTestSupport(ruleContext);
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
  public String addCoverageSupport(JavaCompilationHelper helper, Artifact executable) {
    // This method can be called only for *_binary/*_test targets.
    Preconditions.checkNotNull(executable);
    helper.addCoverageSupport();

    // We do not add the instrumented jar to the runtime classpath, but provide it in the shell
    // script via an environment variable.
    return JACOCO_COVERAGE_RUNNER_MAIN_CLASS;
  }

  @Override
  public CustomCommandLine buildSingleJarCommandLine(
      String toolchainIdentifier,
      Artifact output,
      Label label,
      String mainClass,
      ImmutableList<String> manifestLines,
      Iterable<Artifact> buildInfoFiles,
      ImmutableList<Artifact> resources,
      NestedSet<Artifact> classpath,
      boolean includeBuildData,
      Compression compression,
      Artifact launcher,
      // Explicitly ignoring params since Bazel doesn't yet support one version
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      Artifact oneVersionAllowlistArtifact,
      Artifact sharedArchive,
      boolean multiReleaseDeployJars,
      PathFragment javaHome,
      Artifact libModules,
      NestedSet<Artifact> hermeticInputs,
      NestedSet<String> addExports,
      NestedSet<String> addOpens) {
    return DeployArchiveBuilder.defaultSingleJarCommandLineWithoutOneVersion(
            output,
            label,
            mainClass,
            manifestLines,
            buildInfoFiles,
            resources,
            classpath,
            includeBuildData,
            compression,
            launcher,
            /* multiReleaseDeployJars= */ multiReleaseDeployJars,
            javaHome,
            libModules,
            hermeticInputs,
            addExports,
            addOpens)
        .build();
  }

  @Override
  public void addArtifactToJavaTargetAttribute(JavaTargetAttributes.Builder builder,
      Artifact srcArtifact) {
  }

  @Override
  public PathFragment getDefaultJavaResourcePath(PathFragment path) {
    // Look for src/.../resources to match Maven repository structure.
    List<String> segments = path.splitToListOfSegments();
    for (int i = 0; i < segments.size() - 2; ++i) {
      if (segments.get(i).equals("src") && segments.get(i + 2).equals("resources")) {
        return path.subFragment(i + 3);
      }
    }
    PathFragment javaPath = JavaUtil.getJavaPath(path);
    return javaPath == null ? path : javaPath;
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
  public void setLintProgressMessage(SpawnAction.Builder spawnAction) {
    spawnAction.setProgressMessage("Running Android Lint for: %{label}");
  }
}

