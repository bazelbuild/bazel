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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.ComputedSubstitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.bazel.rules.BazelConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaUtil;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Semantics for Bazel Java rules
 */
public class BazelJavaSemantics implements JavaSemantics {

  public static final BazelJavaSemantics INSTANCE = new BazelJavaSemantics();

  private static final Template STUB_SCRIPT =
      Template.forResource(BazelJavaSemantics.class, "java_stub_template.txt");
  private static final Template STUB_SCRIPT_WINDOWS =
      Template.forResource(BazelJavaSemantics.class, "java_stub_template_windows.txt");

  private static final String JAVABUILDER_CLASS_NAME =
      "com.google.devtools.build.buildjar.BazelJavaBuilder";

  private static final String JACOCO_COVERAGE_RUNNER_MAIN_CLASS =
      "com.google.testing.coverage.JacocoCoverageRunner";
  private static final String BAZEL_TEST_RUNNER_MAIN_CLASS =
      "com.google.testing.junit.runner.BazelTestRunner";
  private static final String EXPERIMENTAL_TEST_RUNNER_MAIN_CLASS =
      "com.google.testing.junit.runner.ExperimentalTestRunner";
  private static final String EXPERIMENTAL_TESTRUNNER_TAG = "experimental_testrunner";

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

  @Override
  public void checkProtoDeps(
      RuleContext ruleContext, Collection<? extends TransitiveInfoCollection> deps) {}

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
        return useExperimentalTestRunner(ruleContext)
            ? EXPERIMENTAL_TEST_RUNNER_MAIN_CLASS
            : BAZEL_TEST_RUNNER_MAIN_CLASS;
      }
    }
    return mainClass;
  }

  private static boolean useExperimentalTestRunner(RuleContext ruleContext) {
    List<String> tags = ruleContext.attributes().get("tags", Type.STRING_LIST);
    return tags.contains(EXPERIMENTAL_TESTRUNNER_TAG);
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
        String key, NestedSet<Artifact> jars, String workspacePrefix, boolean isRunfilesEnabled) {
      super(key);
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
              new PathFragment(new PathFragment(workspacePrefix), artifact.getRunfilesPath());
          buffer.append(runfilePath.normalize().getPathString());
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

  @Override
  public Artifact createStubAction(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact executable,
      String javaStartClass,
      String javaExecutable) {
    Preconditions.checkState(ruleContext.getConfiguration().hasFragment(Jvm.class));

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
    arguments.add(Substitution.of("%javabin%", javaExecutable));
    arguments.add(Substitution.of("%needs_runfiles%",
        ruleContext.getFragment(Jvm.class).getJavaExecutable().isAbsolute() ? "0" : "1"));

    NestedSet<Artifact> classpath = javaCommon.getRuntimeClasspath();
    arguments.add(
        new ComputedClasspathSubstitution(
            "%classpath%", classpath, workspacePrefix, isRunfilesEnabled));

    JavaCompilationArtifacts javaArtifacts = javaCommon.getJavaCompilationArtifacts();
    String path =
        javaArtifacts.getInstrumentedJar() != null
            ? "${JAVA_RUNFILES}/"
                + workspacePrefix
                + javaArtifacts.getInstrumentedJar().getRootRelativePath().getPathString()
            : "";
    arguments.add(
        Substitution.of(
            "%set_jacoco_metadata%",
            ruleContext.getConfiguration().isCodeCoverageEnabled()
                ? "export JACOCO_METADATA_JAR=" + path
                : ""));

    arguments.add(Substitution.of("%java_start_class%",
        ShellEscaper.escapeString(javaStartClass)));
    arguments.add(Substitution.ofSpaceSeparatedList("%jvm_flags%", ImmutableList.copyOf(jvmFlags)));

    ruleContext.registerAction(new TemplateExpansionAction(
        ruleContext.getActionOwner(), executable, STUB_SCRIPT, arguments, true));
    if (OS.getCurrent() == OS.WINDOWS) {
      Artifact newExecutable =
          ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".cmd");
      ruleContext.registerAction(
          new TemplateExpansionAction(
              ruleContext.getActionOwner(),
              newExecutable,
              STUB_SCRIPT_WINDOWS,
              ImmutableList.of(
                  Substitution.of(
                      "%bash_exe_path%",
                      ruleContext
                          .getFragment(BazelConfiguration.class)
                          .getShellExecutable()
                          .getPathString()),
                  Substitution.of(
                      "%cygpath_exe_path%",
                      ruleContext
                          .getFragment(BazelConfiguration.class)
                          .getShellExecutable()
                          .replaceName("cygpath.exe")
                          .getPathString())),
              true));
      return newExecutable;
    } else {
      return executable;
    }
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
          useExperimentalTestRunner(ruleContext) ? "$experimental_testsupport" : "$testsupport";
      return Iterables.getOnlyElement(ruleContext.getPrerequisites(testSupport, Mode.TARGET));
    } else {
      return null;
    }
  }

  @Override
  public void addRunfilesForBinary(RuleContext ruleContext, Artifact launcher,
      Runfiles.Builder runfilesBuilder) {
    TransitiveInfoCollection testSupport = getTestSupport(ruleContext);
    if (testSupport != null) {
      runfilesBuilder.addTarget(testSupport, JavaRunfilesProvider.TO_RUNFILES);
      runfilesBuilder.addTarget(testSupport, RunfilesProvider.DEFAULT_RUNFILES);
    }
  }

  @Override
  public void addRunfilesForLibrary(RuleContext ruleContext, Runfiles.Builder runfilesBuilder) {
  }

  @Override
  public void collectTargetsTreatedAsDeps(
      RuleContext ruleContext, ImmutableList.Builder<TransitiveInfoCollection> builder) {
    TransitiveInfoCollection testSupport = getTestSupport(ruleContext);
    if (testSupport != null) {
      // TODO(bazel-team): The testsupport is used as the test framework
      // and really only needs to be on the runtime, not compile-time
      // classpath.
      builder.add(testSupport);
    }
  }

  @Override
  public Iterable<String> getExtraJavacOpts(RuleContext ruleContext) {
    return ImmutableList.<String>of();
  }

  @Override
  public void addProviders(RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact classJar,
      Artifact srcJar,
      Artifact genJar,
      Artifact gensrcJar,
      ImmutableMap<Artifact, Artifact> compilationToRuntimeJarMap,
      NestedSetBuilder<Artifact> filesBuilder,
      RuleConfiguredTargetBuilder ruleBuilder) {
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

  // TODO(yueg): refactor this (only mainClass different for now)
  @Override
  public String addCoverageSupport(
      JavaCompilationHelper helper,
      JavaTargetAttributes.Builder attributes,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      String mainClass)
      throws InterruptedException {
    // This method can be called only for *_binary/*_test targets.
    Preconditions.checkNotNull(executable);
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

    // Add the coverage runner to the list of dependencies when compiling in coverage mode.
    TransitiveInfoCollection runnerTarget =
        helper.getRuleContext().getPrerequisite("$jacocorunner", Mode.TARGET);
    if (runnerTarget.getProvider(JavaCompilationArgsProvider.class) != null) {
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
  public CustomCommandLine buildSingleJarCommandLine(BuildConfiguration configuration,
      Artifact output, String mainClass, ImmutableList<String> manifestLines,
      Iterable<Artifact> buildInfoFiles, ImmutableList<Artifact> resources,
      Iterable<Artifact> classpath, boolean includeBuildData,
      Compression compression, Artifact launcher) {
    return DeployArchiveBuilder.defaultSingleJarCommandLine(output, mainClass, manifestLines,
        buildInfoFiles, resources, classpath, includeBuildData, compression, launcher).build();
  }

  @Override
  public ImmutableList<Artifact> translate(RuleContext ruleContext, JavaConfiguration javaConfig,
      List<Artifact> messages) {
    return ImmutableList.<Artifact>of();
  }

  @Override
  public Artifact getLauncher(RuleContext ruleContext, JavaCommon common,
      DeployArchiveBuilder deployArchiveBuilder, Runfiles.Builder runfilesBuilder,
      List<String> jvmFlags, JavaTargetAttributes.Builder attributesBuilder, boolean shouldStrip) {
    return JavaHelper.launcherArtifactForTarget(this, ruleContext);
  }

  @Override
  public void addDependenciesForRunfiles(RuleContext ruleContext, Runfiles.Builder builder) {
  }

  @Override
  public void addArtifactToJavaTargetAttribute(JavaTargetAttributes.Builder builder,
      Artifact srcArtifact) {
  }

  @Override
  public void commonDependencyProcessing(RuleContext ruleContext,
      JavaTargetAttributes.Builder attributes,
      Collection<? extends TransitiveInfoCollection> deps) {
  }

  @Override
  public PathFragment getDefaultJavaResourcePath(PathFragment path) {
    // Look for src/.../resources to match Maven repository structure.
    for (int i = 0; i < path.segmentCount() - 2; ++i) {
      if (path.getSegment(i).equals("src") && path.getSegment(i + 2).equals("resources")) {
        return path.subFragment(i + 3, path.segmentCount());
      }
    }
    PathFragment javaPath = JavaUtil.getJavaPath(path);
    return javaPath == null ? path : javaPath;
  }

  @Override
  public List<String> getExtraArguments(RuleContext ruleContext, ImmutableList<Artifact> sources) {
    if (ruleContext.getRule().getRuleClass().equals("java_test")) {
      if (useLegacyJavaTest(ruleContext)) {
        if (ruleContext.getConfiguration().getTestArguments().isEmpty()
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
}
