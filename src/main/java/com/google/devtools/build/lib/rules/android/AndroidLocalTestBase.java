// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.base.Strings.nullToEmpty;
import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression.COMPRESSED;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.SourceManifestAction;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingContext;
import com.google.devtools.build.lib.rules.java.ClasspathConfiguredFragment;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaCompileOutputs;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPrimaryClassProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.JavaOutput;
import com.google.devtools.build.lib.rules.java.JavaRuntimeClasspathProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.OneVersionCheckActionBuilder;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** A base implementation for the "android_local_test" rule. */
public abstract class AndroidLocalTestBase implements RuleConfiguredTargetFactory {

  private final AndroidSemantics androidSemantics;

  protected AndroidLocalTestBase(AndroidSemantics androidSemantics) {
    this.androidSemantics = androidSemantics;
  }

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    ruleContext.checkSrcsSamePackage(true);

    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();

    AndroidDataContext dataContext = androidSemantics.makeContextForNative(ruleContext);
    ResourceApk resourceApk =
        buildResourceApk(
            dataContext,
            androidSemantics,
            ruleContext,
            DataBinding.contextFrom(ruleContext, dataContext.getAndroidConfig()),
            AndroidManifest.fromAttributes(ruleContext, dataContext),
            AndroidResources.from(ruleContext, "resource_files"),
            AndroidAssets.from(ruleContext),
            ResourceDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
            AssetDependencies.fromRuleDeps(ruleContext, /* neverlink = */ false),
            StampedAndroidManifest.getManifestValues(ruleContext),
            ruleContext.getExpander().withDataExecLocations().tokenized("nocompress_extensions"),
            ResourceFilterFactory.fromRuleContextAndAttrs(ruleContext));

    JavaCommon javaCommon =
        AndroidCommon.createJavaCommonWithAndroidDataBinding(
            ruleContext,
            javaSemantics,
            resourceApk.asDataBindingContext(),
            /* isLibrary */ false,
            /* shouldCompileJavaSrcs */ true);
    androidSemantics.checkRule(ruleContext, javaCommon);

    // Use the regular Java javacopts, plus any extra needed for databinding. Enforcing
    // android-compatible Java (-source 7 -target 7 and no TWR) is unnecessary for robolectric tests
    // since they run on a JVM, not an android device.
    JavaToolchainProvider javaToolchain = JavaToolchainProvider.from(ruleContext);
    ImmutableList.Builder<String> javacopts = ImmutableList.builder();
    javacopts.addAll(javaSemantics.getCompatibleJavacOptions(ruleContext, javaToolchain));
    resourceApk
        .asDataBindingContext()
        .supplyJavaCoptsUsing(ruleContext, /* isBinary= */ true, javacopts::addAll);
    JavaTargetAttributes.Builder attributesBuilder =
        javaCommon.initCommon(ImmutableList.of(), javacopts.build());

    resourceApk
        .asDataBindingContext()
        .supplyAnnotationProcessor(
            ruleContext,
            (plugin, additionalOutputs) -> {
              attributesBuilder.addPlugin(plugin);
              attributesBuilder.addAdditionalOutputs(additionalOutputs);
            });

    attributesBuilder.addRuntimeClassPathEntry(resourceApk.getResourceJavaClassJar());

    // Exclude the Rs from the library from the runtime classpath.
    NestedSet<Artifact> excludedRuntimeArtifacts = getLibraryResourceJars(ruleContext);
    attributesBuilder.addExcludedArtifacts(excludedRuntimeArtifacts);

    // Create robolectric test_config.properties file
    String name = "_robolectric/" + ruleContext.getRule().getName() + "_test_config.properties";
    Artifact propertiesFile = ruleContext.getGenfilesArtifact(name);

    String resourcesLocation =
        resourceApk.getValidatedResources().getMergedResources().getRunfilesPathString();
    Template template =
        Template.forResource(AndroidLocalTestBase.class, "robolectric_properties_template.txt");
    List<Substitution> substitutions = new ArrayList<>();
    substitutions.add(
        Substitution.of(
            "%android_merged_manifest%", resourceApk.getManifest().getRunfilesPathString()));
    substitutions.add(
        Substitution.of("%android_merged_resources%", "jar:file:" + resourcesLocation + "!/res"));
    substitutions.add(
        Substitution.of("%android_merged_assets%", "jar:file:" + resourcesLocation + "!/assets"));

    String customPackage = resourceApk.getValidatedResources().getJavaPackage();
    substitutions.add(Substitution.of("%android_custom_package%", nullToEmpty(customPackage)));

    substitutions.add(
        Substitution.of(
            "%android_resource_apk%", resourceApk.getArtifact().getRunfilesPathString()));

    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            propertiesFile,
            template,
            substitutions,
            /* makeExecutable= */ false));
    // Add the properties file to the test jar as a java resource
    attributesBuilder.addResource(
        PathFragment.create("com/android/tools/test_config.properties"), propertiesFile);

    String testClass = getAndCheckTestClass(ruleContext, javaCommon.getSrcsArtifacts());
    getAndCheckTestSupport(ruleContext);
    if (Allowlist.hasAllowlist(ruleContext, "multiple_proto_rule_types_in_deps_allowlist")
        && !Allowlist.isAvailable(ruleContext, "multiple_proto_rule_types_in_deps_allowlist")) {
      javaSemantics.checkForProtoLibraryAndJavaProtoLibraryOnSameProto(ruleContext, javaCommon);
    }
    if (ruleContext.hasErrors()) {
      return null;
    }

    // Databinding metadata that the databinding annotation processor reads.
    ImmutableList<Artifact> additionalJavaInputsFromDatabinding =
        resourceApk.asDataBindingContext().processDeps(ruleContext, /* isBinary= */ true);

    JavaCompilationHelper helper =
        getJavaCompilationHelperWithDependencies(
            ruleContext,
            javaSemantics,
            javaCommon,
            attributesBuilder,
            additionalJavaInputsFromDatabinding);

    Artifact srcJar =
        ruleContext.getImplicitOutputArtifact(AndroidSemantics.ANDROID_BINARY_SOURCE_JAR);
    JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder =
        JavaSourceJarsProvider.builder()
            .addSourceJar(srcJar)
            .addAllTransitiveSourceJars(javaCommon.collectTransitiveSourceJars(srcJar));

    Artifact classJar =
        ruleContext.getImplicitOutputArtifact(AndroidSemantics.ANDROID_BINARY_CLASS_JAR);

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();

    Artifact executable; // the artifact for the rule itself
    if (OS.getCurrent() == OS.WINDOWS) {
      executable =
          ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");
    } else {
      executable = ruleContext.createOutputArtifact();
    }

    String mainClass = androidSemantics.getTestRunnerMainClass();
    String originalMainClass = mainClass;
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      mainClass =
          addCoverageSupport(
              ruleContext,
              javaSemantics,
              helper,
              executable,
              /* instrumentationMetadata= */ null,
              javaArtifactsBuilder,
              attributesBuilder,
              mainClass);
    }

    JavaCompileOutputs<Artifact> outputs = helper.createOutputs(classJar);

    JavaRuleOutputJarsProvider.Builder javaRuleOutputJarsProviderBuilder =
        JavaRuleOutputJarsProvider.builder()
            .addJavaOutput(
                JavaOutput.builder()
                    .fromJavaCompileOutputs(outputs)
                    .setCompileJar(classJar)
                    .setCompileJdeps(
                        javaCommon.getJavaCompilationArtifacts().getCompileTimeDependencyArtifact())
                    .addSourceJar(srcJar)
                    .build());

    javaArtifactsBuilder.setCompileTimeDependencies(outputs.depsProto());

    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().add(classJar).add(executable);

    GeneratedExtensionRegistryProvider generatedExtensionRegistryProvider =
        javaSemantics.createGeneratedExtensionRegistry(
            ruleContext,
            javaCommon,
            filesToBuildBuilder,
            javaArtifactsBuilder,
            javaRuleOutputJarsProviderBuilder,
            javaSourceJarsProviderBuilder);

    JavaTargetAttributes attributes = attributesBuilder.build();
    addJavaClassJarToArtifactsBuilder(javaArtifactsBuilder, attributes, classJar);

    helper.createCompileAction(outputs);
    helper.createSourceJarAction(srcJar, outputs.genSource());

    setUpJavaCommon(javaCommon, helper, javaArtifactsBuilder.build(), attributes);

    Artifact launcher = JavaHelper.launcherArtifactForTarget(javaSemantics, ruleContext);

    String javaExecutable;
    if (javaSemantics.isJavaExecutableSubstitution()) {
      javaExecutable = javaCommon.getJavaBinSubstitution(ruleContext, launcher);
    } else {
      javaExecutable = javaCommon.getJavaExecutableForStub(ruleContext, launcher);
    }

    javaSemantics.createStubAction(
        ruleContext,
        javaCommon,
        getJvmFlags(ruleContext, testClass),
        executable,
        mainClass,
        originalMainClass,
        filesToBuildBuilder,
        javaExecutable,
        /* createCoverageMetadataJar= */ false);

    Artifact oneVersionOutputArtifact = null;
    JavaConfiguration javaConfig = ruleContext.getFragment(JavaConfiguration.class);
    OneVersionEnforcementLevel oneVersionEnforcementLevel = javaConfig.oneVersionEnforcementLevel();

    boolean doOneVersionEnforcement =
        oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF
            && javaConfig.enforceOneVersionOnJavaTests();
    if (doOneVersionEnforcement) {
      oneVersionOutputArtifact =
          OneVersionCheckActionBuilder.newBuilder()
              .withEnforcementLevel(oneVersionEnforcementLevel)
              .useToolchain(javaToolchain)
              .checkJars(
                  NestedSetBuilder.fromNestedSet(attributes.getRuntimeClassPath())
                      .add(classJar)
                      .build())
              .build(ruleContext);
    }

    NestedSet<Artifact> filesToBuild = filesToBuildBuilder.build();

    Runfiles defaultRunfiles =
        collectDefaultRunfiles(
            ruleContext,
            javaCommon,
            filesToBuild,
            resourceApk.getManifest(),
            resourceApk.getResourceJavaClassJar(),
            resourceApk.getValidatedResources().getMergedResources(),
            resourceApk);

    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, defaultRunfiles, executable);

    Artifact deployJar =
        ruleContext.getImplicitOutputArtifact(AndroidSemantics.ANDROID_BINARY_DEPLOY_JAR);

    // Create the deploy jar and make it dependent on the runfiles middleman if an executable is
    // created. Do not add the deploy jar to files to build, so we will only build it when it gets
    // requested.
    new DeployArchiveBuilder(javaSemantics, ruleContext)
        .setOutputJar(deployJar)
        .setJavaStartClass(mainClass)
        .setDeployManifestLines(ImmutableList.of())
        .setAttributes(attributes)
        .addRuntimeJars(javaCommon.getJavaCompilationArtifacts().getRuntimeJars())
        .setIncludeBuildData(true)
        .setCompression(COMPRESSED)
        .setLauncher(launcher)
        .setOneVersionEnforcementLevel(
            doOneVersionEnforcement ? oneVersionEnforcementLevel : OneVersionEnforcementLevel.OFF,
            javaToolchain.getOneVersionAllowlist())
        .build();

    JavaSourceJarsProvider sourceJarsProvider = javaSourceJarsProviderBuilder.build();
    NestedSet<Artifact> transitiveSourceJars = sourceJarsProvider.getTransitiveSourceJars();

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);

    if (generatedExtensionRegistryProvider != null) {
      builder.addNativeDeclaredProvider(generatedExtensionRegistryProvider);
    }

    resourceApk.asDataBindingContext().addProvider(builder, ruleContext);

    JavaRuleOutputJarsProvider ruleOutputJarsProvider = javaRuleOutputJarsProviderBuilder.build();

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();

    ImmutableMap.Builder<String, String> coverageEnvironment = ImmutableMap.builder();
    NestedSetBuilder<Artifact> coverageSupportFiles = NestedSetBuilder.stableOrder();
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {

      // Create an artifact that contains the runfiles relative paths of the jars on the runtime
      // classpath. Using SourceManifestAction is the only reliable way to match the runfiles
      // creation code.
      Artifact runtimeClasspathArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              "runtime_classpath_for_coverage",
              "runtime_classpath.txt",
              ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(
          new SourceManifestAction(
              SourceManifestAction.ManifestType.SOURCES_ONLY,
              ruleContext.getActionOwner(),
              runtimeClasspathArtifact,
              new Runfiles.Builder(
                      ruleContext.getWorkspaceName(),
                      ruleContext.getConfiguration().legacyExternalRunfiles())
                  // This matches the code below in collectDefaultRunfiles.
                  .addTransitiveArtifactsWrappedInStableOrder(javaCommon.getRuntimeClasspath())
                  .build(),
              null,
              true));
      filesToBuildBuilder.add(runtimeClasspathArtifact);

      // Pass the artifact through an environment variable in the coverage environment so it
      // can be read by the coverage collection script.
      coverageEnvironment.put(
          "JAVA_RUNTIME_CLASSPATH_FOR_COVERAGE", runtimeClasspathArtifact.getExecPathString());
      // Add the file to coverageSupportFiles so it ends up as an input for the test action
      // when coverage is enabled.
      coverageSupportFiles.add(runtimeClasspathArtifact);

      // Make single jar reachable from the coverage environment because it needs to be executed
      // by the coverage collection script.
      FilesToRunProvider singleJar = JavaToolchainProvider.from(ruleContext).getSingleJar();
      coverageEnvironment.put("SINGLE_JAR_TOOL", singleJar.getExecutable().getExecPathString());
      coverageSupportFiles.add(singleJar.getExecutable());
    }

    javaCommon.addTransitiveInfoProviders(
        builder,
        javaInfoBuilder,
        filesToBuild,
        classJar,
        coverageEnvironment.build(),
        coverageSupportFiles.build());
    javaCommon.addGenJarsProvider(
        builder, javaInfoBuilder, outputs.genClass(), outputs.genSource());

    javaCommon.addTransitiveInfoProviders(builder, javaInfoBuilder, filesToBuild, classJar);
    javaCommon.addGenJarsProvider(
        builder, javaInfoBuilder, outputs.genClass(), outputs.genSource());

    // Just confirming that there are no aliases being used here.
    AndroidFeatureFlagSetProvider.getAndValidateFlagMapFromRuleContext(ruleContext);

    if (oneVersionOutputArtifact != null) {
      builder.addOutputGroup(OutputGroupInfo.VALIDATION, oneVersionOutputArtifact);
    }

    JavaInfo javaInfo =
        javaInfoBuilder
            .javaSourceJars(sourceJarsProvider)
            .javaRuleOutputs(ruleOutputJarsProvider)
            .build();

    return builder
        .setFilesToBuild(filesToBuild)
        .addStarlarkDeclaredProvider(javaInfo)
        .addProvider(
            RunfilesProvider.class,
            RunfilesProvider.withData(
                defaultRunfiles,
                new Runfiles.Builder(ruleContext.getWorkspaceName())
                    .merge(runfilesSupport)
                    .build()))
        .setRunfilesSupport(runfilesSupport, executable)
        .addProvider(
            JavaRuntimeClasspathProvider.class,
            new JavaRuntimeClasspathProvider(javaCommon.getRuntimeClasspath()))
        .addProvider(JavaPrimaryClassProvider.class, new JavaPrimaryClassProvider(testClass))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .addOutputGroup(
            JavaSemantics.DIRECT_SOURCE_JARS_OUTPUT_GROUP,
            NestedSetBuilder.wrap(Order.STABLE_ORDER, sourceJarsProvider.getSourceJars()))
        .build();
  }

  @Override
  public final void addRuleImplSpecificRequiredConfigFragments(
      RequiredConfigFragmentsProvider.Builder requiredFragments,
      AttributeMap attributes,
      BuildConfigurationValue configuration) {
    requiredFragments.addStarlarkOptions(AndroidFeatureFlagSetProvider.getFeatureFlags(attributes));
  }

  private static void setUpJavaCommon(
      JavaCommon common,
      JavaCompilationHelper helper,
      JavaCompilationArtifacts javaCompilationArtifacts,
      JavaTargetAttributes attributes)
      throws RuleErrorException {
    common.setJavaCompilationArtifacts(javaCompilationArtifacts);
    common.setClassPathFragment(
        new ClasspathConfiguredFragment(
            common.getJavaCompilationArtifacts(),
            attributes,
            false,
            helper.getBootclasspathOrDefault().bootclasspath()));
  }

  private static void addJavaClassJarToArtifactsBuilder(
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      JavaTargetAttributes attributes,
      Artifact classJar) {
    if (attributes.hasSources() || attributes.hasResources()) {
      // We only want to add a jar to the classpath of a dependent rule if it has content.
      javaArtifactsBuilder.addRuntimeJar(classJar);
    }
  }

  private Runfiles collectDefaultRunfiles(
      RuleContext ruleContext,
      JavaCommon javaCommon,
      NestedSet<Artifact> filesToBuild,
      Artifact manifest,
      Artifact resourcesClassJar,
      Artifact resourcesZip,
      @Nullable ResourceApk resourceApk)
      throws RuleErrorException {

    Runfiles.Builder builder = new Runfiles.Builder(ruleContext.getWorkspaceName());
    builder.addTransitiveArtifacts(filesToBuild);
    builder.addArtifacts(javaCommon.getJavaCompilationArtifacts().getRuntimeJars());

    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);

    ImmutableList<TransitiveInfoCollection> depsForRunfiles =
        ImmutableList.<TransitiveInfoCollection>builder()
            .addAll(ruleContext.getPrerequisites("$robolectric_implicit_classpath"))
            .addAll(ruleContext.getPrerequisites("runtime_deps"))
            .build();

    Artifact androidAllJarsPropertiesFile = getAndroidAllJarsPropertiesFile(ruleContext);
    if (androidAllJarsPropertiesFile != null) {
      builder.addArtifact(androidAllJarsPropertiesFile);
    }

    // runtime jars always in naive link order, incompatible with compile order runfiles.
    builder.addArtifacts(getRuntimeJarsForTargets(getAndCheckTestSupport(ruleContext)).toList());

    builder.addTargets(
        depsForRunfiles,
        RunfilesProvider.DEFAULT_RUNFILES,
        ruleContext.getConfiguration().alwaysIncludeFilesToBuildInData());

    // We assume that the runtime jars will not have conflicting artifacts
    // with the same root relative path
    builder.addTransitiveArtifactsWrappedInStableOrder(javaCommon.getRuntimeClasspath());

    // Add the JDK files from P4 (see java_stub_template.txt).
    builder.addTransitiveArtifacts(
        JavaRuntimeInfo.from(
                ruleContext, javaCommon.getJavaSemantics().getJavaRuntimeToolchainType())
            .javaBaseInputs());
    builder.addArtifact(manifest);
    builder.addArtifact(resourcesClassJar);
    builder.addArtifact(resourcesZip);
    if (resourceApk != null) {
      builder.addArtifact(resourceApk.getArtifact());
    }

    return builder.build();
  }

  private static NestedSet<Artifact> getRuntimeJarsForTargets(TransitiveInfoCollection deps)
      throws RuleErrorException {
    // The dep may be a simple JAR and not a java rule, hence we can't simply do
    // dep.getProvider(JavaCompilationArgsProvider.class).getRecursiveJavaCompilationArgs(),
    // so we reuse the logic within JavaCompilationArgs to handle both scenarios.
    return JavaCompilationArgsProvider.legacyFromTargets(ImmutableList.of(deps)).getRuntimeJars();
  }

  private static String getAndCheckTestClass(
      RuleContext ruleContext, ImmutableList<Artifact> sourceFiles) {
    String testClass =
        ruleContext.getRule().isAttrDefined("test_class", Type.STRING)
            ? ruleContext.attributes().get("test_class", Type.STRING)
            : "";

    if (testClass.isEmpty()) {
      testClass = JavaCommon.determinePrimaryClass(ruleContext, sourceFiles);
      if (testClass == null) {
        ruleContext.ruleError(
            "cannot determine junit.framework.Test class "
                + "(Found no source file '"
                + ruleContext.getTarget().getName()
                + ".java' and package name doesn't include 'java' or 'javatests'. "
                + "You might want to rename the rule or add a 'test_class' "
                + "attribute.)");
        testClass = "";
      }
    }
    return testClass;
  }

  static ResourceApk buildResourceApk(
      AndroidDataContext dataContext,
      AndroidSemantics androidSemantics,
      RuleErrorConsumer errorConsumer,
      DataBindingContext dataBindingContext,
      AndroidManifest manifest,
      AndroidResources resources,
      AndroidAssets assets,
      ResourceDependencies resourceDeps,
      AssetDependencies assetDeps,
      Map<String, String> manifestValues,
      List<String> noCompressExtensions,
      ResourceFilterFactory resourceFilterFactory)
      throws InterruptedException {

    StampedAndroidManifest stamped =
        manifest.mergeWithDeps(
            dataContext,
            androidSemantics,
            errorConsumer,
            resourceDeps,
            manifestValues,
            /* manifestMerger = */ null);

    return ProcessedAndroidData.processLocalTestDataFrom(
            dataContext,
            dataBindingContext,
            stamped,
            manifestValues,
            resources,
            assets,
            resourceDeps,
            assetDeps,
            noCompressExtensions,
            resourceFilterFactory)
        .generateRClass(dataContext);
  }

  private static NestedSet<Artifact> getLibraryResourceJars(RuleContext ruleContext) {
    Iterable<AndroidLibraryResourceClassJarProvider> libraryResourceJarProviders =
        AndroidCommon.getTransitivePrerequisites(
            ruleContext, AndroidLibraryResourceClassJarProvider.PROVIDER);

    NestedSetBuilder<Artifact> libraryResourceJarsBuilder = NestedSetBuilder.naiveLinkOrder();
    for (AndroidLibraryResourceClassJarProvider provider : libraryResourceJarProviders) {
      libraryResourceJarsBuilder.addTransitive(provider.getResourceClassJars());
    }
    return libraryResourceJarsBuilder.build();
  }

  /** Get JavaSemantics */
  protected abstract JavaSemantics createJavaSemantics();

  protected abstract AndroidSemantics createAndroidSemantics();

  /** Set test and robolectric specific jvm flags */
  protected abstract ImmutableList<String> getJvmFlags(RuleContext ruleContext, String testClass)
      throws RuleErrorException, InterruptedException;

  /**
   * Enables coverage support for Android and Java targets: adds instrumented jar to the classpath
   * and modifies main class.
   *
   * @return new main class
   */
  protected abstract String addCoverageSupport(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCompilationHelper helper,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      JavaTargetAttributes.Builder attributesBuilder,
      String mainClass)
      throws InterruptedException, RuleErrorException;

  /** Adds compilation dependencies to the java compilation helper. */
  private JavaCompilationHelper getJavaCompilationHelperWithDependencies(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCommon javaCommon,
      JavaTargetAttributes.Builder javaTargetAttributesBuilder,
      ImmutableList<Artifact> additionalArtifacts)
      throws RuleErrorException {
    JavaCompilationHelper javaCompilationHelper =
        new JavaCompilationHelper(
            ruleContext,
            javaSemantics,
            javaCommon.getJavacOpts(),
            javaTargetAttributesBuilder,
            additionalArtifacts);

    if (ruleContext.isAttrDefined("$junit", BuildType.LABEL)) {
      // JUnit jar must be ahead of android runtime jars since these contain stubbed definitions
      // for framework.junit.* classes which Robolectric does not re-write.
      javaCompilationHelper.addLibrariesToAttributes(ruleContext.getPrerequisites("$junit"));
    }
    // Robolectric jars must be ahead of other potentially conflicting jars
    // (e.g., Android runtime jars) in the classpath to make sure they always take precedence.
    javaCompilationHelper.addLibrariesToAttributes(
        ruleContext.getPrerequisites("$robolectric_implicit_classpath"));

    javaCompilationHelper.addLibrariesToAttributes(
        javaCommon.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY));

    javaCompilationHelper.addLibrariesToAttributes(
        ImmutableList.of(getAndCheckTestSupport(ruleContext)));
    return javaCompilationHelper;
  }

  /** Get the testrunner from the rule */
  protected abstract TransitiveInfoCollection getAndCheckTestSupport(RuleContext ruleContext)
      throws RuleErrorException;

  /** Get the android-all jars properties file from the deps */
  @Nullable
  protected abstract Artifact getAndroidAllJarsPropertiesFile(RuleContext ruleContext)
      throws RuleErrorException;
}
