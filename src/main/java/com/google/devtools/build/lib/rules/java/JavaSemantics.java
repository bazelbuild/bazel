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

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.Runfiles.Builder;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.Substitution.ComputedSubstitution;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.Attribute.LabelListLateBoundDefault;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Pluggable Java compilation semantics. */
public interface JavaSemantics {
  SafeImplicitOutputsFunction JAVA_LIBRARY_CLASS_JAR = fromTemplates("lib%{name}.jar");
  SafeImplicitOutputsFunction JAVA_LIBRARY_SOURCE_JAR = fromTemplates("lib%{name}-src.jar");

  SafeImplicitOutputsFunction JAVA_BINARY_CLASS_JAR = fromTemplates("%{name}.jar");
  SafeImplicitOutputsFunction JAVA_BINARY_SOURCE_JAR = fromTemplates("%{name}-src.jar");

  SafeImplicitOutputsFunction JAVA_BINARY_DEPLOY_JAR = fromTemplates("%{name}_deploy.jar");
  SafeImplicitOutputsFunction JAVA_BINARY_MERGED_JAR = fromTemplates("%{name}_merged.jar");
  SafeImplicitOutputsFunction JAVA_UNSTRIPPED_BINARY_DEPLOY_JAR =
      fromTemplates("%{name}_deploy.jar.unstripped");
  SafeImplicitOutputsFunction JAVA_BINARY_PROGUARD_MAP = fromTemplates("%{name}_proguard.map");
  SafeImplicitOutputsFunction JAVA_BINARY_PROGUARD_PROTO_MAP =
      fromTemplates("%{name}_proguard.pbmap");
  SafeImplicitOutputsFunction JAVA_BINARY_PROGUARD_SEEDS = fromTemplates("%{name}_proguard.seeds");
  SafeImplicitOutputsFunction JAVA_BINARY_PROGUARD_USAGE = fromTemplates("%{name}_proguard.usage");
  SafeImplicitOutputsFunction JAVA_BINARY_PROGUARD_CONFIG =
      fromTemplates("%{name}_proguard.config");
  SafeImplicitOutputsFunction JAVA_ONE_VERSION_ARTIFACT = fromTemplates("%{name}-one-version.txt");
  SafeImplicitOutputsFunction SHARED_ARCHIVE_ARTIFACT = fromTemplates("%{name}.jsa");

  SafeImplicitOutputsFunction JAVA_COVERAGE_RUNTIME_CLASS_PATH_TXT =
      fromTemplates("%{name}-runtime-classpath.txt");

  SafeImplicitOutputsFunction JAVA_BINARY_DEPLOY_SOURCE_JAR =
      fromTemplates("%{name}_deploy-src.jar");

  SafeImplicitOutputsFunction JAVA_TEST_CLASSPATHS_FILE = fromTemplates("%{name}_classpaths_file");
  String TEST_RUNTIME_CLASSPATH_FILE_PLACEHOLDER = "%test_runtime_classpath_file%";

  FileType JAVA_SOURCE = FileType.of(".java");
  FileType JAR = FileType.of(".jar");
  FileType PROPERTIES = FileType.of(".properties");
  FileType SOURCE_JAR = FileType.of(".srcjar");

  /** Label to the Java Toolchain rule. It is resolved from a label given in the java options. */
  String JAVA_TOOLCHAIN_LABEL = "//tools/jdk:toolchain";

  /** The java_toolchain.compatible_javacopts key for Android javacopts */
  public static final String ANDROID_JAVACOPTS_KEY = "android";
  /** The java_toolchain.compatible_javacopts key for proto compilations. */
  public static final String PROTO_JAVACOPTS_KEY = "proto";
  /** The java_toolchain.compatible_javacopts key for testonly compilations. */
  public static final String TESTONLY_JAVACOPTS_KEY = "testonly";

  static Label javaToolchainAttribute(RuleDefinitionEnvironment environment) {
    return environment.getToolsLabel("//tools/jdk:current_java_toolchain");
  }

  /** Name of the output group used for source jars. */
  String SOURCE_JARS_OUTPUT_GROUP = OutputGroupInfo.HIDDEN_OUTPUT_GROUP_PREFIX + "source_jars";

  /** Implementation for the :jvm attribute. */
  static Label jvmAttribute(RuleDefinitionEnvironment env) {
    return env.getToolsLabel("//tools/jdk:current_java_runtime");
  }

  /** Implementation for the :host_jdk attribute. */
  static Label hostJdkAttribute(RuleDefinitionEnvironment env) {
    return env.getToolsLabel("//tools/jdk:current_host_java_runtime");
  }

  /**
   * Implementation for the :java_launcher attribute. Note that the Java launcher is disabled by
   * default, so it returns null for the configuration-independent default value.
   */
  @AutoCodec
  LabelLateBoundDefault<JavaConfiguration> JAVA_LAUNCHER =
      LabelLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          null,
          (rule, attributes, javaConfig) -> {
            // This nullness check is purely for the sake of a test that doesn't bother to include
            // an
            // attribute map when calling this method.
            if (attributes != null) {
              // Don't depend on the launcher if we don't create an executable anyway
              if (attributes.has("create_executable")
                  && !attributes.get("create_executable", Type.BOOLEAN)) {
                return null;
              }

              // use_launcher=False disables the launcher
              if (attributes.has("use_launcher") && !attributes.get("use_launcher", Type.BOOLEAN)) {
                return null;
              }

              // don't read --java_launcher if this target overrides via a launcher attribute
              if (attributes.isAttributeValueExplicitlySpecified("launcher")) {
                return attributes.get("launcher", LABEL);
              }
            }
            return javaConfig.getJavaLauncherLabel();
          });

  @AutoCodec
  LabelListLateBoundDefault<JavaConfiguration> JAVA_PLUGINS =
      LabelListLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          (rule, attributes, javaConfig) -> ImmutableList.copyOf(javaConfig.getPlugins()));

  /** Implementation for the :proguard attribute. */
  @AutoCodec
  LabelLateBoundDefault<JavaConfiguration> PROGUARD =
      LabelLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          null,
          (rule, attributes, javaConfig) -> javaConfig.getProguardBinary());

  @AutoCodec
  LabelListLateBoundDefault<JavaConfiguration> EXTRA_PROGUARD_SPECS =
      LabelListLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          (rule, attributes, javaConfig) ->
              ImmutableList.copyOf(javaConfig.getExtraProguardSpecs()));

  @AutoCodec
  LabelLateBoundDefault<JavaConfiguration> BYTECODE_OPTIMIZER =
      LabelLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          null,
          (rule, attributes, javaConfig) -> {
            // Use a modicum of smarts to avoid implicit dependencies where we don't need them.
            boolean hasProguardSpecs =
                attributes.has("proguard_specs")
                    && !attributes.get("proguard_specs", LABEL_LIST).isEmpty();
            JavaConfiguration.NamedLabel optimizer = javaConfig.getBytecodeOptimizer();
            if (!hasProguardSpecs || !optimizer.label().isPresent()) {
              return null;
            }
            return optimizer.label().get();
          });

  String JACOCO_METADATA_PLACEHOLDER = "%set_jacoco_metadata%";
  String JACOCO_MAIN_CLASS_PLACEHOLDER = "%set_jacoco_main_class%";
  String JACOCO_JAVA_RUNFILES_ROOT_PLACEHOLDER = "%set_jacoco_java_runfiles_root%";

  /** Substitution for exporting the jars needed for jacoco coverage. */
  class ComputedJacocoSubstitution extends ComputedSubstitution {
    private final NestedSet<Artifact> jars;
    private final String pathPrefix;

    public ComputedJacocoSubstitution(NestedSet<Artifact> jars, String workspacePrefix) {
      super(JACOCO_METADATA_PLACEHOLDER);
      this.jars = jars;
      this.pathPrefix = "${JAVA_RUNFILES}/" + workspacePrefix;
    }

    /**
     * Concatenating the root relative paths of the artifacts. Each relative path entry is prepended
     * with "${JAVA_RUNFILES}" and the workspace prefix.
     */
    @Override
    public String getValue() {
      return jars.toList().stream()
          .map(artifact -> pathPrefix + "/" + artifact.getRootRelativePathString())
          .collect(Collectors.joining(File.pathSeparator, "export JACOCO_METADATA_JARS=", ""));
    }
  }

  /**
   * Verifies if the rule contains any errors.
   *
   * <p>Errors should be signaled through {@link RuleContext}.
   */
  void checkRule(RuleContext ruleContext, JavaCommon javaCommon);

  /**
   * Verifies there are no conflicts in protos.
   *
   * <p>Errors should be signaled through {@link RuleContext}.
   */
  void checkForProtoLibraryAndJavaProtoLibraryOnSameProto(
      RuleContext ruleContext, JavaCommon javaCommon);

  /** Returns the main class of a Java binary. */
  String getMainClass(RuleContext ruleContext, ImmutableList<Artifact> srcsArtifacts);

  /**
   * Returns the primary class for a Java binary - either the main class, or, in case of a test, the
   * test class (not the test runner main class).
   */
  String getPrimaryClass(RuleContext ruleContext, ImmutableList<Artifact> srcsArtifacts);

  /**
   * Returns the resources contributed by a Java rule (usually the contents of the {@code resources}
   * attribute)
   */
  ImmutableList<Artifact> collectResources(RuleContext ruleContext);

  String getTestRunnerMainClass();

  /**
   * Constructs the command line to call SingleJar to join all artifacts from {@code classpath}
   * (java code) and {@code resources} into {@code output}.
   */
  CustomCommandLine buildSingleJarCommandLine(
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
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      Artifact oneVersionAllowlistArtifact,
      Artifact sharedArchive);

  /**
   * Creates the action that writes the Java executable stub script.
   *
   * <p>Returns the launcher script artifact. This may or may not be the same as {@code executable},
   * depending on the implementation of this method. If they are the same, then this Artifact should
   * be used when creating both the {@code RunfilesProvider} and the {@code RunfilesSupport}. If
   * they are different, the new value should be used when creating the {@code RunfilesProvider} (so
   * it will be the stub script executed by "bazel run" for example), and the old value should be
   * used when creating the {@code RunfilesSupport} (so the runfiles directory will be named after
   * it).
   *
   * <p>For example on Windows we use a double dispatch approach: the launcher is a batch file (and
   * is created and returned by this method) which shells out to a shell script (the {@code
   * executable} argument).
   *
   * <p>In Blaze, this method considers {@code javaExecutable} as a substitution that can be
   * directly used to replace %javabin% in stub script, but in Bazel this method considers {@code
   * javaExecutable} as a file path for the JVM binary (java).
   */
  Artifact createStubAction(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact executable,
      String javaStartClass,
      String javaExecutable)
      throws InterruptedException;

  /**
   * Same as {@link #createStubAction(RuleContext, JavaCommon, List, Artifact, String, String)}.
   *
   * <p>In *experimental* coverage mode creates a txt file containing the runtime jars names. {@code
   * JacocoCoverageRunner} will use it to retrieve the name of the jars considered for collecting
   * coverage. {@code JacocoCoverageRunner} will *not* collect coverage implicitly for all the
   * runtime jars, only for those that pack a file ending in "-paths-for-coverage.txt".
   *
   * @param createCoverageMetadataJar is false for Java rules and true otherwise (e.g. android)
   */
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
      throws InterruptedException;

  /**
   * Returns true if {@code createStubAction} considers {@code javaExecutable} as a substitution.
   * Returns false if {@code createStubAction} considers {@code javaExecutable} as a file path.
   */
  boolean isJavaExecutableSubstitution();

  /**
   * Returns true if target is a test target, has TestConfiguration, and persistent test runner set.
   *
   * <p>Note that no TestConfiguration implies the TestConfiguration was pruned in some parent of
   * the rule. Therefore, TestTarget not currently being analyzed as part of top-level and thus
   * persistent test runner is not especially relevant.
   */
  static boolean isTestTargetAndPersistentTestRunner(RuleContext ruleContext) {
    if (!ruleContext.isTestTarget()) {
      return false;
    }
    TestConfiguration testConfiguration = ruleContext.getFragment(TestConfiguration.class);
    return testConfiguration != null && testConfiguration.isPersistentTestRunner();
  }

  static Runfiles getTestSupportRunfiles(RuleContext ruleContext) {
    TransitiveInfoCollection testSupport = getTestSupport(ruleContext);
    if (testSupport == null) {
      return Runfiles.EMPTY;
    }

    RunfilesProvider testSupportRunfilesProvider = testSupport.getProvider(RunfilesProvider.class);
    return testSupportRunfilesProvider.getDefaultRunfiles();
  }

  static NestedSet<Artifact> getTestSupportRuntimeClasspath(RuleContext ruleContext) {
    TransitiveInfoCollection testSupport = JavaSemantics.getTestSupport(ruleContext);
    if (testSupport == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    return JavaInfo.getProvider(JavaCompilationArgsProvider.class, testSupport).getRuntimeJars();
  }

  static TransitiveInfoCollection getTestSupport(RuleContext ruleContext) {
    if (!isJavaBinaryOrJavaTest(ruleContext)) {
      return null;
    }

    if (useLegacyJavaTest(ruleContext)) {
      return null;
    }

    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);
    if (createExecutable && ruleContext.attributes().get("use_testrunner", Type.BOOLEAN)) {
      return Iterables.getOnlyElement(ruleContext.getPrerequisites("$testsupport"));
    } else {
      return null;
    }
  }

  static boolean useLegacyJavaTest(RuleContext ruleContext) {
    return !ruleContext.attributes().isAttributeValueExplicitlySpecified("test_class")
        && ruleContext.getFragment(JavaConfiguration.class).useLegacyBazelJavaTest();
  }

  static boolean isJavaBinaryOrJavaTest(RuleContext ruleContext) {
    return ruleContext.getRule().getRuleClass().equals("java_binary")
        || ruleContext.getRule().getRuleClass().equals("java_test");
  }

  /** Adds extra runfiles for a {@code java_binary} rule. */
  void addRunfilesForBinary(
      RuleContext ruleContext, Artifact launcher, Runfiles.Builder runfilesBuilder);

  /** Adds extra runfiles for a {@code java_library} rule. */
  void addRunfilesForLibrary(RuleContext ruleContext, Runfiles.Builder runfilesBuilder);

  /**
   * Returns the command line options to be used when compiling Java code for {@code java_*} rules.
   *
   * <p>These will come after the default options specified by the toolchain, and before the ones in
   * the {@code javacopts} attribute.
   */
  ImmutableList<String> getCompatibleJavacOptions(
      RuleContext ruleContext, JavaToolchainProvider toolchain);

  /** Add additional targets to be treated as direct dependencies. */
  void collectTargetsTreatedAsDeps(
      RuleContext ruleContext,
      ImmutableList.Builder<TransitiveInfoCollection> builder,
      ClasspathType type);

  /**
   * Enables coverage support for the java target - adds instrumented jar to the classpath and
   * modifies main class.
   *
   * @return new main class
   */
  String addCoverageSupport(JavaCompilationHelper helper, Artifact executable)
      throws InterruptedException;

  /** Return the JVM flags to be used in a Java binary. */
  Iterable<String> getJvmFlags(
      RuleContext ruleContext, ImmutableList<Artifact> srcsArtifacts, List<String> userJvmFlags);

  /**
   * Adds extra providers to a Java target.
   *
   * @throws InterruptedException
   */
  void addProviders(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      Artifact gensrcJar,
      RuleConfiguredTargetBuilder ruleBuilder)
      throws InterruptedException;

  /** Translates XMB messages to translations artifact suitable for Java targets. */
  ImmutableList<Artifact> translate(
      RuleContext ruleContext, JavaConfiguration javaConfig, List<Artifact> messages);

  /**
   * Get the launcher artifact for a java binary, creating the necessary actions for it.
   *
   * @param ruleContext The rule context
   * @param common The common helper class.
   * @param deployArchiveBuilder the builder to construct the deploy archive action (mutable).
   * @param unstrippedDeployArchiveBuilder the builder to construct the unstripped deploy archive
   *     action (mutable).
   * @param runfilesBuilder the builder to construct the list of runfiles (mutable).
   * @param jvmFlags the list of flags to pass to the JVM when running the Java binary (mutable).
   * @param attributesBuilder the builder to construct the list of attributes of this target
   *     (mutable).
   * @param ccToolchain to be used to build the launcher
   * @param featureConfiguration to be used to configure the cc toolchain
   * @return the launcher and unstripped launcher as an artifact pair. If shouldStrip is false, then
   *     they will be the same.
   * @throws InterruptedException
   */
  Pair<Artifact, Artifact> getLauncher(
      final RuleContext ruleContext,
      final JavaCommon common,
      DeployArchiveBuilder deployArchiveBuilder,
      DeployArchiveBuilder unstrippedDeployArchiveBuilder,
      Builder runfilesBuilder,
      List<String> jvmFlags,
      JavaTargetAttributes.Builder attributesBuilder,
      boolean shouldStrip,
      CcToolchainProvider ccToolchain,
      FeatureConfiguration featureConfiguration)
      throws InterruptedException, RuleErrorException;

  /**
   * Add a source artifact to a {@link JavaTargetAttributes.Builder}. It is called when a source
   * artifact is processed but is not matched by default patterns in the {@link
   * JavaTargetAttributes.Builder#addSourceArtifacts(Iterable)} method. The semantics can then
   * detect its custom artifact types and add it to the builder.
   */
  void addArtifactToJavaTargetAttribute(JavaTargetAttributes.Builder builder, Artifact srcArtifact);

  /**
   * Takes the path of a Java resource and tries to determine the Java root relative path of the
   * resource.
   *
   * <p>This is only used if the Java rule doesn't have a {@code resource_strip_prefix} attribute.
   *
   * @param path the root relative path of the resource.
   * @return the Java root relative path of the resource of the root relative path of the resource
   *     if no Java root relative path can be determined.
   */
  PathFragment getDefaultJavaResourcePath(PathFragment path);

  /** @return a list of extra arguments to appends to the runfiles support. */
  List<String> getExtraArguments(RuleContext ruleContext, ImmutableList<Artifact> sources);

  /** @return main class (entry point) for the Java compiler. */
  String getJavaBuilderMainClass();

  /**
   * @return An artifact representing the protobuf-format version of the proguard mapping, or null
   *     if the proguard version doesn't support this.
   */
  Artifact getProtoMapping(RuleContext ruleContext) throws InterruptedException;

  /**
   * Produces the proto generated extension registry artifacts, or <tt>null</tt> if no registry
   * needs to be generated for the provided <tt>ruleContext</tt>.
   */
  @Nullable
  GeneratedExtensionRegistryProvider createGeneratedExtensionRegistry(
      RuleContext ruleContext,
      JavaCommon common,
      NestedSetBuilder<Artifact> filesBuilder,
      JavaCompilationArtifacts.Builder javaCompilationArtifactsBuilder,
      JavaRuleOutputJarsProvider.Builder javaRuleOutputJarsProviderBuilder,
      JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder)
      throws InterruptedException;

  Artifact getObfuscatedConstantStringMap(RuleContext ruleContext) throws InterruptedException;

  void checkDependencyRuleKinds(RuleContext ruleContext);
}
