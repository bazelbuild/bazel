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
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.Attribute.LabelListLateBoundDefault;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import javax.annotation.Nullable;

/** Pluggable Java compilation semantics. */
public interface JavaSemantics {
  SafeImplicitOutputsFunction JAVA_LIBRARY_CLASS_JAR = fromTemplates("lib%{name}.jar");
  SafeImplicitOutputsFunction JAVA_LIBRARY_SOURCE_JAR = fromTemplates("lib%{name}-src.jar");

  SafeImplicitOutputsFunction JAVA_BINARY_CLASS_JAR = fromTemplates("%{name}.jar");
  SafeImplicitOutputsFunction JAVA_BINARY_SOURCE_JAR = fromTemplates("%{name}-src.jar");

  SafeImplicitOutputsFunction JAVA_BINARY_DEPLOY_JAR = fromTemplates("%{name}_deploy.jar");
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

  SafeImplicitOutputsFunction JAVA_BINARY_DEPLOY_SOURCE_JAR =
      fromTemplates("%{name}_deploy-src.jar");

  FileType JAVA_SOURCE = FileType.of(".java");
  FileType JAR = FileType.of(".jar");
  FileType PROPERTIES = FileType.of(".properties");
  FileType SOURCE_JAR = FileType.of(".srcjar");

  /** The java_toolchain.compatible_javacopts key for Android javacopts */
  public static final String ANDROID_JAVACOPTS_KEY = "android";
  /** The java_toolchain.compatible_javacopts key for testonly compilations. */
  public static final String TESTONLY_JAVACOPTS_KEY = "testonly";

  /** The java_toolchain.compatible_javacopts key for public visibility. */
  public static final String PUBLIC_VISIBILITY_JAVACOPTS_KEY = "public_visibility";

  static Label javaToolchainAttribute(RuleDefinitionEnvironment environment) {
    return environment.getToolsLabel("//tools/jdk:current_java_toolchain");
  }

  /** Name of the output group used for transitive source jars. */
  String SOURCE_JARS_OUTPUT_GROUP = OutputGroupInfo.HIDDEN_OUTPUT_GROUP_PREFIX + "source_jars";

  /** Name of the output group used for direct source jars. */
  String DIRECT_SOURCE_JARS_OUTPUT_GROUP =
      OutputGroupInfo.HIDDEN_OUTPUT_GROUP_PREFIX + "direct_source_jars";

  public String getJavaToolchainType();

  /** Implementation for the :java_launcher attribute. */
  static LabelLateBoundDefault<JavaConfiguration> javaLauncherAttribute(Label defaultLabel) {
    return LabelLateBoundDefault.fromTargetConfiguration(
        JavaConfiguration.class,
        defaultLabel,
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
  }

  @SerializationConstant
  LabelListLateBoundDefault<JavaConfiguration> JAVA_PLUGINS =
      LabelListLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          (rule, attributes, javaConfig) -> ImmutableList.copyOf(javaConfig.getPlugins()));

  /** Implementation for the :proguard attribute. */
  @SerializationConstant
  LabelLateBoundDefault<JavaConfiguration> PROGUARD =
      LabelLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          null,
          (rule, attributes, javaConfig) -> javaConfig.getProguardBinary());

  @SerializationConstant
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
            if ((!hasProguardSpecs && !javaConfig.runLocalJavaOptimizations())
                || !optimizer.label().isPresent()) {
              return null;
            }
            return optimizer.label().get();
          });

  @SerializationConstant
  LabelListLateBoundDefault<JavaConfiguration> LOCAL_JAVA_OPTIMIZATION_CONFIGURATION =
      LabelListLateBoundDefault.fromTargetConfiguration(
          JavaConfiguration.class,
          (rule, attributes, javaConfig) -> {
            // Don't bother adding the configuration dep if we're not going to use it.
            if (!javaConfig.runLocalJavaOptimizations()) {
              return ImmutableList.of();
            }
            return javaConfig.getLocalJavaOptimizationConfiguration();
          });

  String JACOCO_METADATA_PLACEHOLDER = "%set_jacoco_metadata%";
  String JACOCO_MAIN_CLASS_PLACEHOLDER = "%set_jacoco_main_class%";
  String JACOCO_JAVA_RUNFILES_ROOT_PLACEHOLDER = "%set_jacoco_java_runfiles_root%";

  /**
   * Verifies if the rule contains any errors.
   *
   * <p>Errors should be signaled through {@link RuleContext}.
   */
  void checkRule(RuleContext ruleContext, JavaCommon javaCommon) throws RuleErrorException;

  /**
   * Verifies there are no conflicts in protos.
   *
   * <p>Errors should be signaled through {@link RuleContext}.
   */
  void checkForProtoLibraryAndJavaProtoLibraryOnSameProto(
      RuleContext ruleContext, JavaCommon javaCommon);

  /**
   * Returns the resources contributed by a Java rule (usually the contents of the {@code resources}
   * attribute)
   */
  ImmutableList<Artifact> collectResources(RuleContext ruleContext) throws RuleErrorException;

  String getTestRunnerMainClass();

  /**
   * Constructs the command line to call SingleJar to join all artifacts from {@code classpath}
   * (java code) and {@code resources} into {@code output}.
   */
  CustomCommandLine buildSingleJarCommandLine(
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
      OneVersionEnforcementLevel oneVersionEnforcementLevel,
      Artifact oneVersionAllowlistArtifact,
      Artifact sharedArchive,
      boolean multiReleaseDeployJars,
      PathFragment javaHome,
      Artifact libModules,
      NestedSet<Artifact> hermeticInputs,
      NestedSet<String> addExports,
      NestedSet<String> addOpens);

  ImmutableList<Artifact> getBuildInfo(RuleContext ruleContext, int stamp)
      throws RuleErrorException, InterruptedException;

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
      throws InterruptedException, RuleErrorException;

  /**
   * Returns true if {@code createStubAction} considers {@code javaExecutable} as a substitution.
   * Returns false if {@code createStubAction} considers {@code javaExecutable} as a file path.
   */
  boolean isJavaExecutableSubstitution();

  @Nullable
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
      throws InterruptedException, RuleErrorException;

  Artifact getObfuscatedConstantStringMap(RuleContext ruleContext) throws InterruptedException;

  /** Sets the progress message on the lint build action. */
  void setLintProgressMessage(SpawnAction.Builder spawnAction);
}
