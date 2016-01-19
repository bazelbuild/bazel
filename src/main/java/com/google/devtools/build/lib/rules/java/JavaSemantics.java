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

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.LanguageDependentFragment.LibraryLanguage;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.Runfiles.Builder;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabelList;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Collection;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Pluggable Java compilation semantics.
 */
public interface JavaSemantics {

  LibraryLanguage LANGUAGE = new LibraryLanguage("Java");

  SafeImplicitOutputsFunction JAVA_LIBRARY_CLASS_JAR =
      fromTemplates("lib%{name}.jar");
  SafeImplicitOutputsFunction JAVA_LIBRARY_SOURCE_JAR =
      fromTemplates("lib%{name}-src.jar");

  SafeImplicitOutputsFunction JAVA_BINARY_CLASS_JAR =
      fromTemplates("%{name}.jar");
  SafeImplicitOutputsFunction JAVA_BINARY_SOURCE_JAR =
      fromTemplates("%{name}-src.jar");

  SafeImplicitOutputsFunction JAVA_BINARY_DEPLOY_JAR =
      fromTemplates("%{name}_deploy.jar");
  SafeImplicitOutputsFunction JAVA_BINARY_MERGED_JAR =
      fromTemplates("%{name}_merged.jar");
  SafeImplicitOutputsFunction JAVA_UNSTRIPPED_BINARY_DEPLOY_JAR =
      fromTemplates("%{name}_deploy.jar.unstripped");
  SafeImplicitOutputsFunction JAVA_BINARY_PROGUARD_MAP =
      fromTemplates("%{name}_proguard.map");

  SafeImplicitOutputsFunction JAVA_BINARY_DEPLOY_SOURCE_JAR =
      fromTemplates("%{name}_deploy-src.jar");

  FileType JAVA_SOURCE = FileType.of(".java");
  FileType JAR = FileType.of(".jar");
  FileType PROPERTIES = FileType.of(".properties");
  FileType SOURCE_JAR = FileType.of(".srcjar");
  // TODO(bazel-team): Rename this metadata extension to something meaningful.
  FileType COVERAGE_METADATA = FileType.of(".em");

  /**
   * Label to the Java Toolchain rule. It is resolved from a label given in the java options.
   */
  String JAVA_TOOLCHAIN_LABEL = "//tools/defaults:java_toolchain";

  LateBoundLabel<BuildConfiguration> JAVA_TOOLCHAIN =
      new LateBoundLabel<BuildConfiguration>(JAVA_TOOLCHAIN_LABEL, JavaConfiguration.class) {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(JavaConfiguration.class).getToolchainLabel();
        }
      };

  /**
   * Name of the output group used for source jars.
   */
  String SOURCE_JARS_OUTPUT_GROUP =
      OutputGroupProvider.HIDDEN_OUTPUT_GROUP_PREFIX + "source_jars";

  /**
   * Name of the output group used for gen jars (the jars containing the class files for sources
   * generated from annotation processors).
   */
  String GENERATED_JARS_OUTPUT_GROUP =
      OutputGroupProvider.HIDDEN_OUTPUT_GROUP_PREFIX + "gen_jars";

  /**
   * Label of a pseudo-filegroup that contains the boot-classpath entries.
   */
  String JAVAC_BOOTCLASSPATH_LABEL = "//tools/defaults:javac_bootclasspath";

  /**
   * Label of the javac extdir used for compiling Java source code.
   */
  String JAVAC_EXTDIR_LABEL = "//tools/defaults:javac_extdir";

  /**
   * Label of the JavaBuilder JAR used for compiling Java source code.
   */
  String JAVABUILDER_LABEL = "//tools/defaults:javabuilder";

  /**
   * Label of the SingleJar JAR used for creating deploy jars.
   */
  String SINGLEJAR_LABEL = "//tools/defaults:singlejar";

  /**
   * Label of the GenClass JAR used for creating the jar for classes from sources generated from
   * annotation processors.
   */
  String GENCLASS_LABEL = "//tools/defaults:genclass";

  /**
   * Label of pseudo-cc_binary that tells Blaze a java target's JAVABIN is never to be replaced by
   * the contents of --java_launcher; only the JDK's launcher will ever be used.
   */
  Label JDK_LAUNCHER_LABEL = Label.parseAbsoluteUnchecked(
      Constants.TOOLS_REPOSITORY + "//third_party/java/jdk:jdk_launcher");

  /**
   * Implementation for the :jvm attribute.
   */
  LateBoundLabel<BuildConfiguration> JVM =
      new LateBoundLabel<BuildConfiguration>(JavaImplicitAttributes.JDK_LABEL, Jvm.class) {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(Jvm.class).getJvmLabel();
        }
      };

  /**
   * Implementation for the :host_jdk attribute.
   */
  LateBoundLabel<BuildConfiguration> HOST_JDK =
      new LateBoundLabel<BuildConfiguration>(JavaImplicitAttributes.JDK_LABEL, Jvm.class) {
        @Override
        public boolean useHostConfiguration() {
          return true;
        }

        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(Jvm.class).getJvmLabel();
        }
      };

  /**
   * Implementation for the :java_launcher attribute. Note that the Java launcher is disabled by
   * default, so it returns null for the configuration-independent default value.
   */
  LateBoundLabel<BuildConfiguration> JAVA_LAUNCHER =
      new LateBoundLabel<BuildConfiguration>(JavaConfiguration.class) {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(JavaConfiguration.class).getJavaLauncherLabel();
        }
      };

  LateBoundLabelList<BuildConfiguration> JAVA_PLUGINS =
      new LateBoundLabelList<BuildConfiguration>() {
        @Override
        public List<Label> getDefault(Rule rule, BuildConfiguration configuration) {
          return ImmutableList.copyOf(configuration.getPlugins());
        }
      };

  /**
   * Implementation for the :proguard attribute.
   */
  LateBoundLabel<BuildConfiguration> PROGUARD =
      new LateBoundLabel<BuildConfiguration>(JavaConfiguration.class) {
        @Override
        public Label getDefault(Rule rule, BuildConfiguration configuration) {
          return configuration.getFragment(JavaConfiguration.class).getProguardBinary();
        }
      };

  LateBoundLabelList<BuildConfiguration> EXTRA_PROGUARD_SPECS =
      new LateBoundLabelList<BuildConfiguration>() {
        @Override
        public List<Label> getDefault(Rule rule, BuildConfiguration configuration) {
          return ImmutableList.copyOf(
              configuration.getFragment(JavaConfiguration.class).getExtraProguardSpecs());
        }
      };

  String IJAR_LABEL = "//tools/defaults:ijar";

  /**
   * Verifies if the rule contains and errors.
   *
   * <p>Errors should be signaled through {@link RuleContext}.
   */
  void checkRule(RuleContext ruleContext, JavaCommon javaCommon);

  /**
   * Returns the main class of a Java binary.
   */
  String getMainClass(RuleContext ruleContext, JavaCommon javaCommon);

  /**
   * Returns the resources contributed by a Java rule (usually the contents of the
   * {@code resources} attribute)
   */
  ImmutableList<Artifact> collectResources(RuleContext ruleContext);

  /**
   * Creates the instrumentation metadata artifact for the specified output .jar .
   */
  @Nullable
  Artifact createInstrumentationMetadataArtifact(RuleContext ruleContext, Artifact outputJar);

  /**
   * Returns the instrumentation libraries (jars) for the given context.
   */
  Iterable<Artifact> getInstrumentationJars(RuleContext context);

  /**
   * May add extra command line options to the Java compile command line.
   */
  void buildJavaCommandLine(Collection<Artifact> outputs, BuildConfiguration configuration,
      CustomCommandLine.Builder result);


  /**
   * Constructs the command line to call SingleJar to join all artifacts from
   * {@code classpath} (java code) and {@code resources} into {@code output}.
   */
  CustomCommandLine buildSingleJarCommandLine(BuildConfiguration configuration,
      Artifact output, String mainClass, ImmutableList<String> manifestLines,
      Iterable<Artifact> buildInfoFiles, ImmutableList<Artifact> resources,
      Iterable<Artifact> classpath, boolean includeBuildData,
      Compression compression, Artifact launcher);

  /**
   * Creates the action that writes the Java executable stub script.
   */
  void createStubAction(RuleContext ruleContext, final JavaCommon javaCommon,
      List<String> jvmFlags, Artifact executable, String javaStartClass,
      String javaExecutable);

  /**
   * Adds extra runfiles for a {@code java_binary} rule.
   */
  void addRunfilesForBinary(RuleContext ruleContext, Artifact launcher,
      Runfiles.Builder runfilesBuilder);

  /**
   * Adds extra runfiles for a {@code java_library} rule.
   */
  void addRunfilesForLibrary(RuleContext ruleContext, Runfiles.Builder runfilesBuilder);

  /**
   * Returns the additional options to be passed to javac.
   */
  Iterable<String> getExtraJavacOpts(RuleContext ruleContext);

  /**
   * Add additional targets to be treated as direct dependencies.
   */
  void collectTargetsTreatedAsDeps(
      RuleContext ruleContext, ImmutableList.Builder<TransitiveInfoCollection> builder);

  /**
   * Enables coverage support for the java target - adds instrumented jar to the classpath and
   * modifies main class.
   *
   * @return new main class
   */
  String addCoverageSupport(JavaCompilationHelper helper,
      JavaTargetAttributes.Builder attributes,
      Artifact executable, Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder, String mainClass);

  /**
   * Return the JVM flags to be used in a Java binary.
   */
  Iterable<String> getJvmFlags(
      RuleContext ruleContext, JavaCommon javaCommon, List<String> userJvmFlags);

  /**
   * Adds extra providers to a Java target.
   * @throws InterruptedException
   */
  void addProviders(RuleContext ruleContext,
      JavaCommon javaCommon,
      List<String> jvmFlags,
      Artifact classJar,
      Artifact srcJar,
      Artifact genJar,
      Artifact gensrcJar,
      ImmutableMap<Artifact, Artifact> compilationToRuntimeJarMap,
      JavaCompilationHelper helper,
      NestedSetBuilder<Artifact> filesBuilder,
      RuleConfiguredTargetBuilder ruleBuilder) throws InterruptedException;

  /**
   * Translates XMB messages to translations artifact suitable for Java targets.
   */
  Collection<Artifact> translate(RuleContext ruleContext, JavaConfiguration javaConfig,
      List<Artifact> messages);

  /**
   * Get the launcher artifact for a java binary, creating the necessary actions for it.
   *
   * @param ruleContext The rule context
   * @param common The common helper class.
   * @param deployArchiveBuilder the builder to construct the deploy archive action (mutable).
   * @param runfilesBuilder the builder to construct the list of runfiles (mutable).
   * @param jvmFlags the list of flags to pass to the JVM when running the Java binary (mutable).
   * @param attributesBuilder the builder to construct the list of attributes of this target
   *        (mutable).
   * @return the launcher as an artifact
   * @throws InterruptedException
   */
  Artifact getLauncher(final RuleContext ruleContext, final JavaCommon common,
      DeployArchiveBuilder deployArchiveBuilder, Runfiles.Builder runfilesBuilder,
      List<String> jvmFlags, JavaTargetAttributes.Builder attributesBuilder, boolean shouldStrip)
      throws InterruptedException;

  /**
   * Add extra dependencies for runfiles of a Java binary.
   */
  void addDependenciesForRunfiles(RuleContext ruleContext, Builder builder);

  /**
   * Determines if we should enforce the use of the :java_launcher target to determine the java
   * launcher artifact even if the --java_launcher option was not specified.
   */
  boolean forceUseJavaLauncherTarget(RuleContext ruleContext);

  /**
   * Add a source artifact to a {@link JavaTargetAttributes.Builder}. It is called when a source
   * artifact is processed but is not matched by default patterns in the
   * {@link JavaTargetAttributes.Builder#addSourceArtifacts(Iterable)} method. The semantics can
   * then detect its custom artifact types and add it to the builder.
   */
  void addArtifactToJavaTargetAttribute(JavaTargetAttributes.Builder builder, Artifact srcArtifact);

  /**
   * Works on the list of dependencies of a java target to builder the {@link JavaTargetAttributes}.
   * This work is performed in {@link JavaCommon} for all java targets.
   */
  void commonDependencyProcessing(RuleContext ruleContext, JavaTargetAttributes.Builder attributes,
      Collection<? extends TransitiveInfoCollection> deps);

  /**
   * Takes the path of a Java resource and tries to determine the Java
   * root relative path of the resource.
   *
   * <p>This is only used if the Java rule doesn't have a {@code resource_strip_prefix} attribute.
   *
   * @param path the root relative path of the resource.
   * @return the Java root relative path of the resource of the root
   *         relative path of the resource if no Java root relative path can be
   *         determined.
   */
  PathFragment getDefaultJavaResourcePath(PathFragment path);

  /**
   * @return a list of extra arguments to appends to the runfiles support.
   */
  List<String> getExtraArguments(RuleContext ruleContext, JavaCommon javaCommon);

  /**
   * @return main class (entry point) for the Java compiler.
   */
  String getJavaBuilderMainClass();
}
