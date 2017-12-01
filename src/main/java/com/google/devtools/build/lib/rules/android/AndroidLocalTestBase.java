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
// limitations under the License.package com.google.devtools.build.lib.rules.android;
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Substitution;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionAction.Template;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.java.ClasspathConfiguredFragment;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaPrimaryClassProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeClasspathProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.OneVersionCheckActionBuilder;
import com.google.devtools.build.lib.rules.java.ProguardHelper;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/** A base implementation for the "android_local_test" rule. */
public abstract class AndroidLocalTestBase implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    ruleContext.checkSrcsSamePackage(true);

    JavaSemantics javaSemantics = createJavaSemantics();
    AndroidSemantics androidSemantics = createAndroidSemantics();

    final JavaCommon javaCommon = new JavaCommon(ruleContext, javaSemantics);
    // Use the regular Java javacopts. Enforcing android-compatible Java
    // (-source 7 -target 7 and no TWR) is unnecessary for robolectric tests
    // since they run on a JVM, not an android device.
    JavaTargetAttributes.Builder attributesBuilder =
        getJavaTargetAttributes(ruleContext, javaCommon);

    // Create the final merged manifest
    ResourceDependencies resourceDependencies = getResourceDependencies(ruleContext);
    ApplicationManifest applicationManifest =
        getApplicationManifest(ruleContext, androidSemantics, resourceDependencies);
    Artifact manifest = applicationManifest.getManifest();
    String manifestLocation = manifest.getRunfilesPathString();

    // Create merged resources and assets zip file
    Artifact resourcesZip =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP);
    String resourcesLocation = resourcesZip.getRunfilesPathString();

    // Create the final merged R class
    Artifact resourcesClassJar =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR);
    ResourceApk resourceApk =
        applicationManifest.packBinaryWithDataAndResources(
            ruleContext,
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_APK),
            resourceDependencies,
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
            ResourceFilterFactory.fromRuleContext(ruleContext),
            ImmutableList.of(), /* list of uncompressed extensions */
            false, /* crunch png */
            ProguardHelper.getProguardConfigArtifact(ruleContext, ""),
            null, /* mainDexProguardCfg */
            false, /* conditionalKeepRules */
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
            resourcesZip,
            DataBinding.isEnabled(ruleContext) ? DataBinding.getLayoutInfoFile(ruleContext) : null,
            null, /* featureOfArtifact */
            null /* featureAfterArtifact */);
    compileResourceJar(ruleContext, resourceApk, resourcesClassJar);
    attributesBuilder.addRuntimeClassPathEntry(resourcesClassJar);

    // Exclude the Rs from the library from the runtime classpath.
    NestedSet<Artifact> excludedRuntimeArtifacts = getLibraryResourceJars(ruleContext);
    attributesBuilder.addExcludedArtifacts(excludedRuntimeArtifacts);

    // Create robolectric test_config.properties file
    String name = "_robolectric/" + ruleContext.getRule().getName() + "_test_config.properties";
    Artifact propertiesFile = ruleContext.getGenfilesArtifact(name);

    Template template =
        Template.forResource(AndroidLocalTestBase.class, "robolectric_properties_template.txt");
    List<Substitution> substitutions = new ArrayList<>();
    substitutions.add(Substitution.of("%android_merged_manifest%", manifestLocation));
    substitutions.add(
        Substitution.of("%android_merged_resources%", "jar:file:" + resourcesLocation + "!/res"));
    substitutions.add(
        Substitution.of("%android_merged_assets%", "jar:file:" + resourcesLocation + "!/assets"));
    substitutions.add(
        Substitution.of(
            "%android_custom_package%", resourceApk.getPrimaryResource().getJavaPackage()));

    ruleContext.registerAction(
        new TemplateExpansionAction(
            ruleContext.getActionOwner(),
            propertiesFile,
            template,
            substitutions,
            false /* makeExecutable */));
    // Add the properties file to the test jar as a java resource
    attributesBuilder.addResource(
        PathFragment.create("com/android/tools/test_config.properties"), propertiesFile);

    String testClass =
        getAndCheckTestClass(ruleContext, ImmutableList.copyOf(attributesBuilder.getSourceFiles()));
    getAndCheckTestSupport(ruleContext);
    javaSemantics.checkForProtoLibraryAndJavaProtoLibraryOnSameProto(ruleContext, javaCommon);
    if (ruleContext.hasErrors()) {
      return null;
    }

    Artifact srcJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_SOURCE_JAR);
    JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder =
        JavaSourceJarsProvider.builder()
            .addSourceJar(srcJar)
            .addAllTransitiveSourceJars(javaCommon.collectTransitiveSourceJars(srcJar));

    Artifact classJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_CLASS_JAR);
    JavaRuleOutputJarsProvider.Builder javaRuleOutputJarsProviderBuilder =
        JavaRuleOutputJarsProvider.builder()
            .addOutputJar(
                classJar,
                classJar,
                srcJar == null ? ImmutableList.<Artifact>of() : ImmutableList.of(srcJar));

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();
    JavaCompilationHelper helper =
        getJavaCompilationHelperWithDependencies(
            ruleContext, javaSemantics, javaCommon, attributesBuilder);
    Artifact instrumentationMetadata =
        helper.createInstrumentationMetadata(classJar, javaArtifactsBuilder);
    Artifact executable; // the artifact for the rule itself
    if (OS.getCurrent() == OS.WINDOWS
        && ruleContext.getConfiguration().enableWindowsExeLauncher()) {
      executable =
          ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");
    } else {
      executable = ruleContext.createOutputArtifact();
    }
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

    String mainClass =
        getMainClass(
            ruleContext,
            javaSemantics,
            helper,
            executable,
            instrumentationMetadata,
            javaArtifactsBuilder,
            attributesBuilder);

    // JavaCompilationHelper.getAttributes() builds the JavaTargetAttributes, after which the
    // JavaTargetAttributes becomes immutable. This is an extra safety check to avoid inconsistent
    // states (i.e. building the JavaTargetAttributes then modifying it again).
    addJavaClassJarToArtifactsBuilder(javaArtifactsBuilder, helper.getAttributes(), classJar);

    // The gensrc jar is created only if the target uses annotation processing. Otherwise,
    // it is null, and the source jar action will not depend on the compile action.
    Artifact manifestProtoOutput = helper.createManifestProtoOutput(classJar);

    Artifact genClassJar = null;
    Artifact genSourceJar = null;
    if (helper.usesAnnotationProcessing()) {
      genClassJar = helper.createGenJar(classJar);
      genSourceJar = helper.createGensrcJar(classJar);
      helper.createGenJarAction(classJar, manifestProtoOutput, genClassJar);
    }
    Artifact outputDepsProtoArtifact =
        helper.createOutputDepsProtoArtifact(classJar, javaArtifactsBuilder);
    javaRuleOutputJarsProviderBuilder.setJdeps(outputDepsProtoArtifact);

    helper.createCompileAction(
        classJar,
        manifestProtoOutput,
        genSourceJar,
        outputDepsProtoArtifact,
        instrumentationMetadata);
    helper.createSourceJarAction(srcJar, genSourceJar);

    setUpJavaCommon(javaCommon, helper, javaArtifactsBuilder.build());

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
        javaExecutable);

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
                  NestedSetBuilder.fromNestedSet(helper.getAttributes().getRuntimeClassPath())
                      .add(classJar)
                      .build())
              .build(ruleContext);
    }

    NestedSet<Artifact> filesToBuild = filesToBuildBuilder.build();

    Runfiles defaultRunfiles =
        collectDefaultRunfiles(
            ruleContext, javaCommon, filesToBuild, manifest, resourcesClassJar, resourcesZip);

    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(
            ruleContext, defaultRunfiles, executable, CustomCommandLine.EMPTY);

    JavaSourceJarsProvider sourceJarsProvider = javaSourceJarsProviderBuilder.build();
    NestedSet<Artifact> transitiveSourceJars = sourceJarsProvider.getTransitiveSourceJars();

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);

    if (generatedExtensionRegistryProvider != null) {
      builder.addProvider(
          GeneratedExtensionRegistryProvider.class, generatedExtensionRegistryProvider);
    }

    JavaRuleOutputJarsProvider ruleOutputJarsProvider = javaRuleOutputJarsProviderBuilder.build();

    javaCommon.addTransitiveInfoProviders(builder, filesToBuild, classJar);
    javaCommon.addGenJarsProvider(builder, genClassJar, genSourceJar);

    // Just confirming that there are no aliases being used here.
    AndroidFeatureFlagSetProvider.getAndValidateFlagMapFromRuleContext(ruleContext);

    if (oneVersionOutputArtifact != null) {
      builder.addOutputGroup(OutputGroupProvider.HIDDEN_TOP_LEVEL, oneVersionOutputArtifact);
    }

    NestedSet<Artifact> extraFilesToRun =
        NestedSetBuilder.create(Order.STABLE_ORDER, runfilesSupport.getRunfilesMiddleman());

    JavaInfo javaInfo =
        JavaInfo.Builder.create()
            .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
            .addProvider(JavaRuleOutputJarsProvider.class, ruleOutputJarsProvider)
            .build();

    return builder
        .setFilesToBuild(filesToBuild)
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
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
        .addProvider(
            JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(helper.getAttributes(), javaSemantics))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .build();
  }

  /** Creates the final merged R class with all of the transitive resources. */
  private void compileResourceJar(
      RuleContext ruleContext, ResourceApk resourceApk, Artifact resourceClassJar)
      throws InterruptedException, RuleErrorException {

    if (resourceApk.getResourceJavaClassJar() == null) {
      new RClassGeneratorActionBuilder(ruleContext)
          .targetAaptVersion(AndroidAaptVersion.chooseTargetAaptVersion(ruleContext))
          .withPrimary(resourceApk.getPrimaryResource())
          .withDependencies(resourceApk.getResourceDependencies())
          .setClassJarOut(resourceClassJar)
          .build();
    } else {
      Preconditions.checkArgument(resourceApk.getResourceJavaClassJar().equals(resourceClassJar));
    }
  }

  /**
   * Returns a merged {@link ApplicationManifest} for the rule. The final merged manifest will be
   * merged into the manifest provided on the rule, or into a placeholder manifest if one is not
   * provided
   * @throws InterruptedException
   * @throws RuleErrorException
   */
  private ApplicationManifest getApplicationManifest(
      RuleContext ruleContext,
      AndroidSemantics androidSemantics,
      ResourceDependencies resourceDependencies)
      throws InterruptedException, RuleErrorException {
    ApplicationManifest applicationManifest;

    if (LocalResourceContainer.definesAndroidResources(ruleContext.attributes())) {
      LocalResourceContainer.validateRuleContext(ruleContext);
      ApplicationManifest ruleManifest = androidSemantics.getManifestForRule(ruleContext);
      applicationManifest = ruleManifest.mergeWith(ruleContext, resourceDependencies, false);
    } else {
      // we don't have a manifest, merge like android_library with a stub manifest
      ApplicationManifest dummyManifest = ApplicationManifest.generatedManifest(ruleContext);
      applicationManifest = dummyManifest.mergeWith(ruleContext, resourceDependencies, false);
    }
    return applicationManifest;
  }

  /** Returns the transitive closure of resource dependencies, including those specified on the rule
   * if present. */
  private ResourceDependencies getResourceDependencies(RuleContext ruleContext) {
    return LocalResourceContainer.definesAndroidResources(ruleContext.attributes())
        ? ResourceDependencies.fromRuleDeps(ruleContext, false /* neverlink */)
        : ResourceDependencies.fromRuleResourceAndDeps(ruleContext, false /* neverlink */);
  }

  private static void setUpJavaCommon(
      JavaCommon common,
      JavaCompilationHelper helper,
      JavaCompilationArtifacts javaCompilationArtifacts) {
    common.setJavaCompilationArtifacts(javaCompilationArtifacts);
    common.setClassPathFragment(
        new ClasspathConfiguredFragment(
            common.getJavaCompilationArtifacts(),
            helper.getAttributes(),
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
      Artifact resourcesZip)
      throws RuleErrorException {

    Runfiles.Builder builder = new Runfiles.Builder(ruleContext.getWorkspaceName());
    builder.addTransitiveArtifacts(filesToBuild);
    builder.addArtifacts(javaCommon.getJavaCompilationArtifacts().getRuntimeJars());

    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    builder.add(ruleContext, JavaRunfilesProvider.TO_RUNFILES);

    List<TransitiveInfoCollection> depsForRunfiles = new ArrayList<>();

    if (ruleContext.isAttrDefined("$robolectric", LABEL_LIST)) {
      depsForRunfiles.addAll(ruleContext.getPrerequisites("$robolectric", Mode.TARGET));
    }
    depsForRunfiles.addAll(ruleContext.getPrerequisites("runtime_deps", Mode.TARGET));

    builder.addArtifacts(getRuntimeJarsForTargets(getAndCheckTestSupport(ruleContext)));

    builder.addTargets(depsForRunfiles, JavaRunfilesProvider.TO_RUNFILES);
    builder.addTargets(depsForRunfiles, RunfilesProvider.DEFAULT_RUNFILES);

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      Artifact instrumentedJar = javaCommon.getJavaCompilationArtifacts().getInstrumentedJar();
      if (instrumentedJar != null) {
        builder.addArtifact(instrumentedJar);
      }
    }

    // We assume that the runtime jars will not have conflicting artifacts
    // with the same root relative path
    builder.addTransitiveArtifactsWrappedInStableOrder(javaCommon.getRuntimeClasspath());

    // Add the JDK files if it comes from P4 (see java_stub_template.txt).
    TransitiveInfoCollection javabaseTarget = ruleContext.getPrerequisite(":jvm", Mode.TARGET);

    if (javabaseTarget != null) {
      builder.addTransitiveArtifacts(
          javabaseTarget.getProvider(FileProvider.class).getFilesToBuild());
    }

    builder.addArtifact(manifest);
    builder.addArtifact(resourcesClassJar);
    builder.addArtifact(resourcesZip);

    return builder.build();
  }

  private NestedSet<Artifact> getRuntimeJarsForTargets(TransitiveInfoCollection deps) {
    // The dep may be a simple JAR and not a java rule, hence we can't simply do
    // dep.getProvider(JavaCompilationArgsProvider.class).getRecursiveJavaCompilationArgs(),
    // so we reuse the logic within JavaCompilationArgs to handle both scenarios.
    JavaCompilationArgs args =
        JavaCompilationArgs.builder()
            .addTransitiveTargets(
                ImmutableList.of(deps), /*recursive=*/ true, ClasspathType.RUNTIME_ONLY)
            .build();
    return args.getRuntimeJars();
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
      }
    }
    return testClass;
  }

  private static NestedSet<Artifact> getLibraryResourceJars(RuleContext ruleContext) {
    Iterable<AndroidLibraryResourceClassJarProvider> libraryResourceJarProviders =
        AndroidCommon.getTransitivePrerequisites(
            ruleContext, Mode.TARGET, AndroidLibraryResourceClassJarProvider.class);

    NestedSetBuilder<Artifact> libraryResourceJarsBuilder = NestedSetBuilder.naiveLinkOrder();
    for (AndroidLibraryResourceClassJarProvider provider : libraryResourceJarProviders) {
      libraryResourceJarsBuilder.addTransitive(provider.getResourceClassJars());
    }
    return libraryResourceJarsBuilder.build();
  }

  /** Get JavaSemantics */
  protected abstract JavaSemantics createJavaSemantics();

  /** Get AndroidSemantics */
  protected abstract AndroidSemantics createAndroidSemantics();

  /** Get JavaTargetAttributes Builder */
  protected abstract JavaTargetAttributes.Builder getJavaTargetAttributes(
      RuleContext ruleContext, JavaCommon javaCommon);

  /** Set test and robolectric specific jvm flags */
  protected abstract ImmutableList<String> getJvmFlags(RuleContext ruleContext, String testClass);

  /** Return the testrunner main class */
  protected abstract String getMainClass(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCompilationHelper helper,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      JavaTargetAttributes.Builder attributesBuilder)
      throws InterruptedException;

  /**
   * Add compilation dependencies to the java compilation helper.
   * @throws RuleErrorException
   */
  protected abstract JavaCompilationHelper getJavaCompilationHelperWithDependencies(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCommon javaCommon,
      JavaTargetAttributes.Builder javaTargetAttributesBuilder)
      throws RuleErrorException;

  /** Get the testrunner from the rule */
  protected abstract TransitiveInfoCollection getAndCheckTestSupport(RuleContext ruleContext)
      throws RuleErrorException;
}
