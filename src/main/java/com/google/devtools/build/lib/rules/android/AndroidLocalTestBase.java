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

import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression.COMPRESSED;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.Substitution;
import com.google.devtools.build.lib.analysis.actions.Template;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
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
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeClasspathProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaStarlarkApiProvider;
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
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    ruleContext.checkSrcsSamePackage(true);

    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();
    AndroidLocalTestConfiguration androidLocalTestConfiguration =
        ruleContext.getFragment(AndroidLocalTestConfiguration.class);

    final JavaCommon javaCommon = new JavaCommon(ruleContext, javaSemantics);
    javaSemantics.checkRule(ruleContext, javaCommon);

    // Use the regular Java javacopts. Enforcing android-compatible Java
    // (-source 7 -target 7 and no TWR) is unnecessary for robolectric tests
    // since they run on a JVM, not an android device.
    JavaTargetAttributes.Builder attributesBuilder = javaCommon.initCommon();

    final AndroidDataContext dataContext = androidSemantics.makeContextForNative(ruleContext);
    final ResourceApk resourceApk =
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
    substitutions.add(
        Substitution.of(
            "%android_custom_package%", resourceApk.getValidatedResources().getJavaPackage()));

    boolean generateBinaryResources =
        androidLocalTestConfiguration.useAndroidLocalTestBinaryResources();
    if (generateBinaryResources) {
      substitutions.add(
          Substitution.of(
              "%android_resource_apk%", resourceApk.getArtifact().getRunfilesPathString()));
    }

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

    JavaCompilationHelper helper =
        getJavaCompilationHelperWithDependencies(
            ruleContext, javaSemantics, javaCommon, attributesBuilder);

    Artifact srcJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_SOURCE_JAR);
    JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder =
        JavaSourceJarsProvider.builder()
            .addSourceJar(srcJar)
            .addAllTransitiveSourceJars(javaCommon.collectTransitiveSourceJars(srcJar));

    Artifact classJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_CLASS_JAR);

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();

    Artifact executable; // the artifact for the rule itself
    if (OS.getCurrent() == OS.WINDOWS) {
      executable =
          ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");
    } else {
      executable = ruleContext.createOutputArtifact();
    }

    String mainClass = javaSemantics.getTestRunnerMainClass();
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
            .addOutputJar(
                classJar,
                classJar,
                outputs.manifestProto(),
                srcJar == null ? ImmutableList.<Artifact>of() : ImmutableList.of(srcJar));

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

    javaRuleOutputJarsProviderBuilder.setJdeps(outputs.depsProto());
    helper.createCompileAction(outputs);
    helper.createSourceJarAction(srcJar, outputs.genSource());

    setUpJavaCommon(javaCommon, helper, javaArtifactsBuilder.build(), attributes);

    Artifact launcher = JavaHelper.launcherArtifactForTarget(javaSemantics, ruleContext);

    String javaExecutable;
    if (javaSemantics.isJavaExecutableSubstitution()) {
      javaExecutable = JavaCommon.getJavaBinSubstitution(ruleContext, launcher);
    } else {
      javaExecutable = JavaCommon.getJavaExecutableForStub(ruleContext, launcher);
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
        /* createCoverageMetadataJar= */ true);

    Artifact oneVersionOutputArtifact = null;
    JavaConfiguration javaConfig = ruleContext.getFragment(JavaConfiguration.class);
    OneVersionEnforcementLevel oneVersionEnforcementLevel = javaConfig.oneVersionEnforcementLevel();

    boolean doOneVersionEnforcement =
        oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF
            && javaConfig.enforceOneVersionOnJavaTests();
    JavaToolchainProvider javaToolchain = JavaToolchainProvider.from(ruleContext);
    if (doOneVersionEnforcement) {
      oneVersionOutputArtifact =
          OneVersionCheckActionBuilder.newBuilder()
              .withEnforcementLevel(oneVersionEnforcementLevel)
              .outputArtifact(
                  ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_ONE_VERSION_ARTIFACT))
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
            generateBinaryResources ? resourceApk : null);

    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(ruleContext, defaultRunfiles, executable);

    Artifact deployJar =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_DEPLOY_JAR);

    // Create the deploy jar and make it dependent on the runfiles middleman if an executable is
    // created. Do not add the deploy jar to files to build, so we will only build it when it gets
    // requested.
    new DeployArchiveBuilder(javaSemantics, ruleContext)
        .setOutputJar(deployJar)
        .setJavaStartClass(mainClass)
        .setDeployManifestLines(ImmutableList.<String>of())
        .setAttributes(attributes)
        .addRuntimeJars(javaCommon.getJavaCompilationArtifacts().getRuntimeJars())
        .setIncludeBuildData(true)
        .setRunfilesMiddleman(runfilesSupport.getRunfilesMiddleman())
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

    JavaRuleOutputJarsProvider ruleOutputJarsProvider = javaRuleOutputJarsProviderBuilder.build();

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();

    javaCommon.addTransitiveInfoProviders(builder, javaInfoBuilder, filesToBuild, classJar);
    javaCommon.addGenJarsProvider(
        builder, javaInfoBuilder, outputs.genClass(), outputs.genSource());

    // Just confirming that there are no aliases being used here.
    AndroidFeatureFlagSetProvider.getAndValidateFlagMapFromRuleContext(ruleContext);
    // Report set feature flags as required "config fragments".
    // While these aren't technically fragments, in practice they're user-defined settings with
    // the same meaning: pieces of configuration the rule requires to work properly. So it makes
    // sense to treat them equivalently for "requirements" reporting purposes.
    builder.addRequiredConfigFragments(AndroidFeatureFlagSetProvider.getFlagNames(ruleContext));

    if (oneVersionOutputArtifact != null) {
      builder.addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, oneVersionOutputArtifact);
    }

    NestedSet<Artifact> extraFilesToRun =
        NestedSetBuilder.create(Order.STABLE_ORDER, runfilesSupport.getRunfilesMiddleman());

    JavaInfo javaInfo =
        javaInfoBuilder
            .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
            .addProvider(JavaRuleOutputJarsProvider.class, ruleOutputJarsProvider)
            .addProvider(
                JavaSourceInfoProvider.class,
                JavaSourceInfoProvider.fromJavaTargetAttributes(attributes, javaSemantics))
            .build();

    return builder
        .setFilesToBuild(filesToBuild)
        .addStarlarkTransitiveInfo(
            JavaStarlarkApiProvider.NAME, JavaStarlarkApiProvider.fromRuleContext())
        .addNativeDeclaredProvider(javaInfo)
        .addProvider(
            RunfilesProvider.class,
            RunfilesProvider.withData(
                defaultRunfiles,
                new Runfiles.Builder(ruleContext.getWorkspaceName())
                    .merge(runfilesSupport)
                    .build()))
        .addFilesToRun(extraFilesToRun)
        .setRunfilesSupport(runfilesSupport, executable)
        .addProvider(
            JavaRuntimeClasspathProvider.class,
            new JavaRuntimeClasspathProvider(javaCommon.getRuntimeClasspath()))
        .addProvider(JavaPrimaryClassProvider.class, new JavaPrimaryClassProvider(testClass))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .build();
  }

  private static void setUpJavaCommon(
      JavaCommon common,
      JavaCompilationHelper helper,
      JavaCompilationArtifacts javaCompilationArtifacts,
      JavaTargetAttributes attributes) {
    common.setJavaCompilationArtifacts(javaCompilationArtifacts);
    common.setClassPathFragment(
        new ClasspathConfiguredFragment(
            common.getJavaCompilationArtifacts(),
            attributes,
            false,
            helper.getBootclasspathOrDefault()));
  }

  private void addJavaClassJarToArtifactsBuilder(
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
    builder.add(ruleContext, JavaRunfilesProvider.TO_RUNFILES);

    ImmutableList<TransitiveInfoCollection> depsForRunfiles =
        ImmutableList.<TransitiveInfoCollection>builder()
            .addAll(ruleContext.getPrerequisites("$robolectric_implicit_classpath"))
            .addAll(ruleContext.getPrerequisites("runtime_deps"))
            .build();

    Artifact androidAllJarsPropertiesFile = getAndroidAllJarsPropertiesFile(ruleContext);
    if (androidAllJarsPropertiesFile != null) {
      builder.addArtifact(androidAllJarsPropertiesFile);
    }

    builder.addArtifacts(getRuntimeJarsForTargets(getAndCheckTestSupport(ruleContext)));

    builder.addTargets(depsForRunfiles, JavaRunfilesProvider.TO_RUNFILES);
    builder.addTargets(depsForRunfiles, RunfilesProvider.DEFAULT_RUNFILES);

    // We assume that the runtime jars will not have conflicting artifacts
    // with the same root relative path
    builder.addTransitiveArtifactsWrappedInStableOrder(javaCommon.getRuntimeClasspath());

    // Add the JDK files from P4 (see java_stub_template.txt).
    builder.addTransitiveArtifacts(JavaRuntimeInfo.from(ruleContext).javaBaseInputs());
    builder.addArtifact(manifest);
    builder.addArtifact(resourcesClassJar);
    builder.addArtifact(resourcesZip);
    if (resourceApk != null) {
      builder.addArtifact(resourceApk.getArtifact());
    }

    return builder.build();
  }

  private NestedSet<Artifact> getRuntimeJarsForTargets(TransitiveInfoCollection deps) {
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
      throws RuleErrorException;

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

  /**
   * Add compilation dependencies to the java compilation helper.
   *
   * @throws RuleErrorException
   */
  private JavaCompilationHelper getJavaCompilationHelperWithDependencies(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCommon javaCommon,
      JavaTargetAttributes.Builder javaTargetAttributesBuilder)
      throws RuleErrorException {
    JavaCompilationHelper javaCompilationHelper =
        new JavaCompilationHelper(
            ruleContext, javaSemantics, javaCommon.getJavacOpts(), javaTargetAttributesBuilder);

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
