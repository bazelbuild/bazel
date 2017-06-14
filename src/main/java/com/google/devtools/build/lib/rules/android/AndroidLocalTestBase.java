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
import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression.COMPRESSED;
import static com.google.devtools.build.lib.vfs.FileSystemUtils.replaceExtension;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarProvider.Aar;
import com.google.devtools.build.lib.rules.java.ClasspathConfiguredFragment;
import com.google.devtools.build.lib.rules.java.DeployArchiveBuilder;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaCompilationHelper;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaPrimaryClassProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRunfilesProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeClasspathProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaTargetAttributes;
import com.google.devtools.build.lib.rules.java.OneVersionCheckActionBuilder;
import com.google.devtools.build.lib.rules.java.SingleJarActionBuilder;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * An base implementation for the "android_local_test" rule.
 */
public abstract class AndroidLocalTestBase implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {

    ruleContext.checkSrcsSamePackage(true);

    JavaSemantics javaSemantics = createJavaSemantics();

    final JavaCommon javaCommon = new JavaCommon(ruleContext, javaSemantics);
    // Use the regular Java javacopts. Enforcing android-compatible Java
    // (-source 7 -target 7 and no TWR) is unnecessary for robolectric tests
    // since they run on a JVM, not an android device.
    JavaTargetAttributes.Builder attributesBuilder = javaCommon.initCommon();

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
        getJavaCompilationHelperWithDependencies(ruleContext, javaSemantics, javaCommon,
            attributesBuilder);
    Artifact instrumentationMetadata =
        helper.createInstrumentationMetadata(classJar, javaArtifactsBuilder);
    Artifact executable = ruleContext.createOutputArtifact(); // the artifact for the rule itself
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
    addRuntimeJarsToArtifactsBuilder(javaArtifactsBuilder, helper.getAttributes(), classJar);

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

    javaSemantics.createStubAction(
        ruleContext,
        javaCommon,
        getJvmFlags(ruleContext, testClass),
        executable,
        mainClass,
        JavaCommon.getJavaBinSubstitution(ruleContext, launcher));

    Artifact deployJar =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_DEPLOY_JAR);

    OneVersionEnforcementLevel oneVersionEnforcementLevel =
        ruleContext.getFragment(JavaConfiguration.class).oneVersionEnforcementLevel();
    if (oneVersionEnforcementLevel != OneVersionEnforcementLevel.OFF) {
      Artifact oneVersionOutput =
          ruleContext
              .getAnalysisEnvironment()
              .getDerivedArtifact(
                  replaceExtension(classJar.getRootRelativePath(), "-one-version.txt"),
                  classJar.getRoot());
      filesToBuildBuilder.add(oneVersionOutput);

      NestedSet<Artifact> transitiveDependencies =
          NestedSetBuilder.fromNestedSet(helper.getAttributes().getRuntimeClassPath())
              .add(classJar)
              .build();
      OneVersionCheckActionBuilder.build(
          ruleContext,
          transitiveDependencies,
          oneVersionOutput,
          oneVersionEnforcementLevel);
    }

    NestedSet<Artifact> filesToBuild = filesToBuildBuilder.build();

    Iterable<AndroidLibraryAarProvider> androidAarProviders =
        Sets.newLinkedHashSet(
            Iterables.concat(
                ruleContext.getPrerequisites(
                    "runtime_deps", Mode.TARGET, AndroidLibraryAarProvider.class),
                ruleContext.getPrerequisites(
                    "deps", Mode.TARGET, AndroidLibraryAarProvider.class)));

    NestedSetBuilder<Aar> transitiveAarsBuilder = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<Aar> strictAarsBuilder = NestedSetBuilder.naiveLinkOrder();
    NestedSetBuilder<Artifact> transitiveAarArtifactsBuilder = NestedSetBuilder.stableOrder();
    for (AndroidLibraryAarProvider aarProvider : androidAarProviders) {
      transitiveAarsBuilder.addTransitive(aarProvider.getTransitiveAars());
      transitiveAarArtifactsBuilder.addTransitive(aarProvider.getTransitiveAarArtifacts());
      if (aarProvider.getAar() != null) {
        strictAarsBuilder.add(aarProvider.getAar());
      }
    }
    NestedSet<Aar> transitiveAars = transitiveAarsBuilder.build();
    NestedSet<Aar> strictAars = strictAarsBuilder.build();
    NestedSet<Artifact> transitiveAarArtifacts = transitiveAarArtifactsBuilder.build();

    Runfiles defaultRunfiles =
        collectDefaultRunfiles(ruleContext, javaCommon, filesToBuild, transitiveAarArtifacts);

    CustomCommandLine.Builder cmdLineArgs = CustomCommandLine.builder();
    if (!transitiveAars.isEmpty()) {
      cmdLineArgs.addJoinValues(
          "--android_libraries",
          ",",
          transitiveAars,
          new Function<Aar, String>() {
            @Override
            public String apply(Aar aar) {
              return aarCmdLineArg(aar);
            }
          });
    }
    if (!strictAars.isEmpty()) {
      cmdLineArgs.addJoinValues(
          "--strict_libraries",
          ",",
          strictAars,
          new Function<Aar, String>() {
            @Override
            public String apply(Aar aar) {
              return aarCmdLineArg(aar);
            }
          });
    }
    RunfilesSupport runfilesSupport =
        RunfilesSupport.withExecutable(
            ruleContext, defaultRunfiles, executable, cmdLineArgs.build());

    // Create the deploy jar and make it dependent on the runfiles middleman if an executable is
    // created. Do not add the deploy jar to files to build, so we will only build it when it gets
    // requested.
    new DeployArchiveBuilder(javaSemantics, ruleContext)
        .setOutputJar(deployJar)
        .setJavaStartClass(mainClass)
        .setDeployManifestLines(ImmutableList.<String>of())
        .setAttributes(helper.getAttributes())
        .addRuntimeJars(javaCommon.getJavaCompilationArtifacts().getRuntimeJars())
        .setIncludeBuildData(true)
        .setRunfilesMiddleman(runfilesSupport.getRunfilesMiddleman())
        .setCompression(COMPRESSED)
        .setLauncher(launcher)
        .build();

    JavaSourceJarsProvider sourceJarsProvider = javaSourceJarsProviderBuilder.build();
    NestedSet<Artifact> transitiveSourceJars = sourceJarsProvider.getTransitiveSourceJars();

    // TODO(bazel-team): if (getOptions().sourceJars) then make this a dummy prerequisite for the
    // DeployArchiveAction ? Needs a few changes there as we can't pass inputs
    SingleJarActionBuilder.createSourceJarAction(
        ruleContext,
        ImmutableMap.<PathFragment, Artifact>of(),
        transitiveSourceJars.toCollection(),
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_DEPLOY_SOURCE_JAR));

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);

    if (generatedExtensionRegistryProvider != null) {
      builder.addProvider(
          GeneratedExtensionRegistryProvider.class,
          generatedExtensionRegistryProvider);
    }

    addExtraProviders(builder, javaCommon, classJar, srcJar, genClassJar, genSourceJar);

    JavaRuleOutputJarsProvider ruleOutputJarsProvider = javaRuleOutputJarsProviderBuilder.build();
    JavaSkylarkApiProvider.Builder skylarkApiProvider =
        JavaSkylarkApiProvider.builder()
            .setRuleOutputJarsProvider(ruleOutputJarsProvider)
            .setSourceJarsProvider(sourceJarsProvider);

    javaCommon.addTransitiveInfoProviders(builder, skylarkApiProvider, filesToBuild, classJar);
    javaCommon.addGenJarsProvider(builder, skylarkApiProvider, genClassJar, genSourceJar);

    // No need to use the flag map here - just confirming that dynamic configurations are in use.
    // TODO(mstaib): remove when static configurations are removed.
    AndroidFeatureFlagSetProvider.getAndValidateFlagMapFromRuleContext(ruleContext);

    return builder
        .setFilesToBuild(filesToBuild)
        .addSkylarkTransitiveInfo(JavaSkylarkApiProvider.NAME, skylarkApiProvider.build())
        .addProvider(ruleOutputJarsProvider)
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
        .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
        .addProvider(JavaPrimaryClassProvider.class, new JavaPrimaryClassProvider(testClass))
        .addProvider(
            JavaSourceInfoProvider.class,
            JavaSourceInfoProvider.fromJavaTargetAttributes(helper.getAttributes(), javaSemantics))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .build();
  }

  private static String aarCmdLineArg(Aar aar) {
    return aar.getManifest().getRootRelativePathString()
        + ":"
        + aar.getAar().getRootRelativePathString();
  }

  protected abstract JavaSemantics createJavaSemantics();

  protected abstract void addExtraProviders(
      RuleConfiguredTargetBuilder builder,
      JavaCommon javaCommon,
      Artifact classJar,
      Artifact srcJar,
      Artifact genClassJar,
      Artifact genSourceJar);

  protected abstract ImmutableList<String> getJvmFlags(RuleContext ruleContext, String testClass);

  protected abstract String getMainClass(
      RuleContext ruleContext,
      JavaSemantics javaSemantics,
      JavaCompilationHelper helper,
      Artifact executable,
      Artifact instrumentationMetadata,
      JavaCompilationArtifacts.Builder javaArtifactsBuilder,
      JavaTargetAttributes.Builder attributesBuilder)
      throws InterruptedException;

  protected abstract JavaCompilationHelper getJavaCompilationHelperWithDependencies(
      RuleContext ruleContext, JavaSemantics javaSemantics, JavaCommon javaCommon,
      JavaTargetAttributes.Builder javaTargetAttributesBuilder);

  protected abstract void getJavaContracts(
      RuleContext ruleContext, List<TransitiveInfoCollection> depsForRunfiles);

  protected static TransitiveInfoCollection getAndCheckTestSupport(RuleContext ruleContext) {
    // Add the unit test support to the list of dependencies.
    TransitiveInfoCollection testSupport = null;
    TransitiveInfoCollection t =
        Iterables.getOnlyElement(ruleContext.getPrerequisites("$testsupport", Mode.TARGET));
    if (t.getProvider(JavaCompilationArgsProvider.class) != null) {
      testSupport = t;
    } else {
      ruleContext.attributeError(
          "$testsupport", "this prerequisite is not a java_library rule, or contains errors");
    }
    return testSupport;
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

  private static void addRuntimeJarsToArtifactsBuilder(
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
      NestedSet<Artifact> transitiveAarArtifacts) {
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

    getJavaContracts(ruleContext, depsForRunfiles);

    depsForRunfiles.add(getAndCheckTestSupport(ruleContext));

    builder.addTargets(depsForRunfiles, JavaRunfilesProvider.TO_RUNFILES);
    builder.addTargets(depsForRunfiles, RunfilesProvider.DEFAULT_RUNFILES);
    builder.addTransitiveArtifacts(transitiveAarArtifacts);

    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      Artifact instrumentedJar = javaCommon.getJavaCompilationArtifacts().getInstrumentedJar();
      if (instrumentedJar != null) {
        builder.addArtifact(instrumentedJar);
      }
    }

    builder.addArtifacts(javaCommon.getRuntimeClasspath());

    // Add the JDK files if it comes from P4 (see java_stub_template.txt).
    TransitiveInfoCollection javabaseTarget = ruleContext.getPrerequisite(":jvm", Mode.TARGET);

    if (javabaseTarget != null) {
      builder.addTransitiveArtifacts(
          javabaseTarget.getProvider(FileProvider.class).getFilesToBuild());
    }
    return builder.build();
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
}
