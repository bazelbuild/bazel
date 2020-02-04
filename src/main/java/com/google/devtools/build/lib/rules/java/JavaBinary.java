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

import static com.google.devtools.build.lib.rules.cpp.CppRuleClasses.STATIC_LINKING_MODE;
import static com.google.devtools.build.lib.rules.java.DeployArchiveBuilder.Compression.COMPRESSED;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.LazyWritePathsFileAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider.ClasspathType;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.OneVersionEnforcementLevel;
import com.google.devtools.build.lib.rules.java.proto.GeneratedExtensionRegistryProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/** An implementation of java_binary. */
public class JavaBinary implements RuleConfiguredTargetFactory {
  private static final PathFragment CPP_RUNTIMES = PathFragment.create("_cpp_runtimes");

  private final JavaSemantics semantics;

  protected JavaBinary(JavaSemantics semantics) {
    this.semantics = semantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    JavaCommon.checkRuleLoadedThroughMacro(ruleContext);
    final JavaCommon common = new JavaCommon(ruleContext, semantics);
    DeployArchiveBuilder deployArchiveBuilder = new DeployArchiveBuilder(semantics, ruleContext);
    Runfiles.Builder runfilesBuilder =
        new Runfiles.Builder(
            ruleContext.getWorkspaceName(),
            ruleContext.getConfiguration().legacyExternalRunfiles());
    List<String> jvmFlags = new ArrayList<>();

    JavaTargetAttributes.Builder attributesBuilder = common.initCommon();
    attributesBuilder.addClassPathResources(
        ruleContext.getPrerequisiteArtifacts("classpath_resources", Mode.TARGET).list());

    // Add Java8 timezone resource data
    addTimezoneResourceForJavaBinaries(ruleContext, attributesBuilder);

    List<String> userJvmFlags = JavaCommon.getJvmFlags(ruleContext);

    ruleContext.checkSrcsSamePackage(true);
    boolean createExecutable = ruleContext.attributes().get("create_executable", Type.BOOLEAN);

    if (!createExecutable) {
      // TODO(cushon): disallow combining launcher=JDK_LAUNCHER_LABEL with create_executable=0
      // and use isAttributeExplicitlySpecified here
      Label launcherAttribute = ruleContext.attributes().get("launcher", BuildType.LABEL);
      if (launcherAttribute != null && !JavaHelper.isJdkLauncher(ruleContext, launcherAttribute)) {
        ruleContext.ruleError("launcher specified but create_executable is false");
      }
    }

    semantics.checkRule(ruleContext, common);
    semantics.checkForProtoLibraryAndJavaProtoLibraryOnSameProto(ruleContext, common);
    String mainClass = semantics.getMainClass(ruleContext, common.getSrcsArtifacts());
    String originalMainClass = mainClass;
    if (ruleContext.hasErrors()) {
      return null;
    }

    // Collect the transitive dependencies.
    JavaCompilationHelper helper =
        new JavaCompilationHelper(ruleContext, semantics, common.getJavacOpts(), attributesBuilder);
    List<TransitiveInfoCollection> deps =
        Lists.newArrayList(common.targetsTreatedAsDeps(ClasspathType.COMPILE_ONLY));
    helper.addLibrariesToAttributes(deps);
    attributesBuilder.addNativeLibraries(
        collectNativeLibraries(common.targetsTreatedAsDeps(ClasspathType.BOTH)));

    // deploy_env is valid for java_binary, but not for java_test.
    if (ruleContext.getRule().isAttrDefined("deploy_env", BuildType.LABEL_LIST)) {
      for (JavaRuntimeClasspathProvider envTarget :
          ruleContext.getPrerequisites(
              "deploy_env", Mode.TARGET, JavaRuntimeClasspathProvider.class)) {
        attributesBuilder.addExcludedArtifacts(envTarget.getRuntimeClasspathNestedSet());
      }
    }

    Artifact srcJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_SOURCE_JAR);
    JavaSourceJarsProvider.Builder javaSourceJarsProviderBuilder =
        JavaSourceJarsProvider.builder()
            .addSourceJar(srcJar)
            .addAllTransitiveSourceJars(common.collectTransitiveSourceJars(srcJar));
    Artifact classJar = ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_CLASS_JAR);

    CppConfiguration cppConfiguration =
        ruleContext.getConfiguration().getFragment(CppConfiguration.class);
    CcToolchainProvider ccToolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(
            ruleContext,
            /* requestedFeatures= */ ImmutableSet.<String>builder()
                .addAll(ruleContext.getFeatures())
                .add(STATIC_LINKING_MODE)
                .build(),
            /* unsupportedFeatures= */ ruleContext.getDisabledFeatures(),
            ccToolchain);
    boolean stripAsDefault =
        ccToolchain.shouldCreatePerObjectDebugInfo(featureConfiguration, cppConfiguration)
            && cppConfiguration.getCompilationMode() == CompilationMode.OPT;
    DeployArchiveBuilder unstrippedDeployArchiveBuilder = null;
    if (stripAsDefault) {
      unstrippedDeployArchiveBuilder = new DeployArchiveBuilder(semantics, ruleContext);
    }
    Pair<Artifact, Artifact> launcherAndUnstrippedLauncher =
        semantics.getLauncher(
            ruleContext,
            common,
            deployArchiveBuilder,
            unstrippedDeployArchiveBuilder,
            runfilesBuilder,
            jvmFlags,
            attributesBuilder,
            stripAsDefault,
            ccToolchain,
            featureConfiguration);
    Artifact launcher = launcherAndUnstrippedLauncher.first;
    Artifact unstrippedLauncher = launcherAndUnstrippedLauncher.second;

    JavaCompilationArtifacts.Builder javaArtifactsBuilder = new JavaCompilationArtifacts.Builder();

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    Artifact executableForRunfiles = null;
    if (createExecutable) {
      // This artifact is named as the rule itself, e.g. //foo:bar_bin -> bazel-bin/foo/bar_bin
      // On Windows, it's going to be bazel-bin/foo/bar_bin.exe
      if (OS.getCurrent() == OS.WINDOWS) {
        executableForRunfiles =
            ruleContext.getImplicitOutputArtifact(ruleContext.getTarget().getName() + ".exe");
      } else {
        executableForRunfiles = ruleContext.createOutputArtifact();
      }
      filesBuilder.add(classJar).add(executableForRunfiles);

      if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
        mainClass = semantics.addCoverageSupport(helper, executableForRunfiles);
      }
    } else {
      filesBuilder.add(classJar);
    }

    JavaCompileOutputs<Artifact> outputs = helper.createOutputs(classJar);
    JavaRuleOutputJarsProvider.Builder ruleOutputJarsProviderBuilder =
        JavaRuleOutputJarsProvider.builder()
            .addOutputJar(
                /* classJar= */ classJar,
                /* iJar= */ null,
                /* manifestProto= */ outputs.manifestProto(),
                /* sourceJars= */ ImmutableList.of(srcJar));

    JavaTargetAttributes attributes = attributesBuilder.build();
    List<Artifact> nativeLibraries = attributes.getNativeLibraries();
    if (!nativeLibraries.isEmpty()) {
      jvmFlags.add(
          "-Djava.library.path="
              + JavaCommon.javaLibraryPath(
                  nativeLibraries, ruleContext.getRule().getPackage().getWorkspaceName()));
    }

    JavaConfiguration javaConfig = ruleContext.getFragment(JavaConfiguration.class);
    if (attributes.hasMessages()) {
      helper.setTranslations(
          semantics.translate(ruleContext, javaConfig, attributes.getMessages()));
    }

    if (attributes.hasSources() || attributes.hasResources()) {
      // We only want to add a jar to the classpath of a dependent rule if it has content.
      javaArtifactsBuilder.addRuntimeJar(classJar);
    }

    GeneratedExtensionRegistryProvider generatedExtensionRegistryProvider =
        semantics.createGeneratedExtensionRegistry(
            ruleContext,
            common,
            filesBuilder,
            javaArtifactsBuilder,
            ruleOutputJarsProviderBuilder,
            javaSourceJarsProviderBuilder);
    javaArtifactsBuilder.setCompileTimeDependencies(outputs.depsProto());
    ruleOutputJarsProviderBuilder.setJdeps(outputs.depsProto());

    JavaCompilationArtifacts javaArtifacts = javaArtifactsBuilder.build();
    common.setJavaCompilationArtifacts(javaArtifacts);

    helper.createCompileAction(outputs);
    helper.createSourceJarAction(srcJar, outputs.genSource());

    common.setClassPathFragment(
        new ClasspathConfiguredFragment(
            javaArtifacts, attributes, false, helper.getBootclasspathOrDefault()));

    Iterables.addAll(
        jvmFlags, semantics.getJvmFlags(ruleContext, common.getSrcsArtifacts(), userJvmFlags));
    if (ruleContext.hasErrors()) {
      return null;
    }

    Artifact executableToRun = executableForRunfiles;
    if (createExecutable) {
      String javaExecutable;
      if (semantics.isJavaExecutableSubstitution()) {
        javaExecutable = JavaCommon.getJavaBinSubstitution(ruleContext, launcher);
      } else {
        javaExecutable = JavaCommon.getJavaExecutableForStub(ruleContext, launcher);
      }
      // Create a shell stub for a Java application
      executableToRun =
          semantics.createStubAction(
              ruleContext,
              common,
              jvmFlags,
              executableForRunfiles,
              mainClass,
              originalMainClass,
              filesBuilder,
              javaExecutable,
              /* createCoverageMetadataJar= */ false);
      if (!executableToRun.equals(executableForRunfiles)) {
        filesBuilder.add(executableToRun);
        runfilesBuilder.addArtifact(executableToRun);
      }
    }

    JavaSourceJarsProvider sourceJarsProvider = javaSourceJarsProviderBuilder.build();
    NestedSet<Artifact> transitiveSourceJars = sourceJarsProvider.getTransitiveSourceJars();

    // TODO(bazel-team): if (getOptions().sourceJars) then make this a dummy prerequisite for the
    // DeployArchiveAction ? Needs a few changes there as we can't pass inputs
    SingleJarActionBuilder.createSourceJarAction(
        ruleContext,
        semantics,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        transitiveSourceJars,
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_DEPLOY_SOURCE_JAR));

    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    builder.add(
        JavaPrimaryClassProvider.class,
        new JavaPrimaryClassProvider(
            semantics.getPrimaryClass(ruleContext, common.getSrcsArtifacts())));
    if (generatedExtensionRegistryProvider != null) {
      builder.addNativeDeclaredProvider(generatedExtensionRegistryProvider);
    }

    Artifact deployJar =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_BINARY_DEPLOY_JAR);

    if (javaConfig.oneVersionEnforcementLevel() != OneVersionEnforcementLevel.OFF) {
      // This JavaBinary class is also the implementation for java_test targets (via the
      // {Google,Bazel}JavaTest subclass). java_test targets can have their one version enforcement
      // disabled with a second flag (to avoid the incremental build performance cost at the expense
      // of safety.)
      if (javaConfig.enforceOneVersionOnJavaTests() || !isJavaTestRule(ruleContext)) {
        builder.addOutputGroup(
            OutputGroupInfo.HIDDEN_TOP_LEVEL,
            OneVersionCheckActionBuilder.newBuilder()
                .withEnforcementLevel(javaConfig.oneVersionEnforcementLevel())
                .outputArtifact(
                    ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_ONE_VERSION_ARTIFACT))
                .useToolchain(JavaToolchainProvider.from(ruleContext))
                .checkJars(
                    NestedSetBuilder.fromNestedSet(attributes.getRuntimeClassPath())
                        .add(classJar)
                        .build())
                .build(ruleContext));
      }
    }
    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    NestedSet<Artifact> dynamicRuntimeActionInputs;
    try {
      dynamicRuntimeActionInputs = ccToolchain.getDynamicRuntimeLinkInputs(featureConfiguration);
    } catch (EvalException e) {
      throw ruleContext.throwWithRuleError(e.getMessage());
    }

    collectDefaultRunfiles(
        runfilesBuilder,
        ruleContext,
        common,
        javaArtifacts,
        filesToBuild,
        launcher,
        dynamicRuntimeActionInputs);
    Runfiles defaultRunfiles = runfilesBuilder.build();

    RunfilesSupport runfilesSupport = null;
    Runfiles persistentTestRunnerRunfiles = null;
    NestedSetBuilder<Artifact> extraFilesToRunBuilder = NestedSetBuilder.stableOrder();
    if (createExecutable) {
      List<String> extraArgs =
          new ArrayList<>(semantics.getExtraArguments(ruleContext, common.getSrcsArtifacts()));
      // The executable we pass here will be used when creating the runfiles directory. E.g. for the
      // stub script called bazel-bin/foo/bar_bin, the runfiles directory will be created under
      // bazel-bin/foo/bar_bin.runfiles . On platforms where there's an extra stub script (Windows)
      // which dispatches to this one, we still create the runfiles directory for the shell script,
      // but use the dispatcher script (a batch file) as the RunfilesProvider's executable.
      runfilesSupport =
          RunfilesSupport.withExecutable(
              ruleContext, defaultRunfiles, executableForRunfiles, extraArgs);
      extraFilesToRunBuilder.add(runfilesSupport.getRunfilesMiddleman());
      if (JavaSemantics.isPersistentTestRunner(ruleContext)) {
        persistentTestRunnerRunfiles = JavaSemantics.getTestSupportRunfiles(ruleContext);
      }
    }

    RunfilesProvider runfilesProvider =
        RunfilesProvider.withData(
            defaultRunfiles,
            new Runfiles.Builder(
                    ruleContext.getWorkspaceName(),
                    ruleContext.getConfiguration().legacyExternalRunfiles())
                .merge(runfilesSupport)
                .build());

    ImmutableList<String> deployManifestLines =
        getDeployManifestLines(ruleContext, originalMainClass);

    deployArchiveBuilder
        .setOutputJar(deployJar)
        .setJavaStartClass(mainClass)
        .setDeployManifestLines(deployManifestLines)
        .setAttributes(attributes)
        .addRuntimeJars(javaArtifacts.getRuntimeJars())
        .setIncludeBuildData(true)
        .setRunfilesMiddleman(
            runfilesSupport == null ? null : runfilesSupport.getRunfilesMiddleman())
        .setCompression(COMPRESSED)
        .setLauncher(launcher)
        .setOneVersionEnforcementLevel(
            javaConfig.oneVersionEnforcementLevel(),
            JavaToolchainProvider.from(ruleContext).getOneVersionWhitelist())
        .build();

    Artifact unstrippedDeployJar =
        ruleContext.getImplicitOutputArtifact(JavaSemantics.JAVA_UNSTRIPPED_BINARY_DEPLOY_JAR);
    if (stripAsDefault) {
      unstrippedDeployArchiveBuilder
          .setOutputJar(unstrippedDeployJar)
          .setJavaStartClass(mainClass)
          .setDeployManifestLines(deployManifestLines)
          .setAttributes(attributes)
          .addRuntimeJars(javaArtifacts.getRuntimeJars())
          .setIncludeBuildData(true)
          .setRunfilesMiddleman(
              runfilesSupport == null ? null : runfilesSupport.getRunfilesMiddleman())
          .setCompression(COMPRESSED)
          .setLauncher(unstrippedLauncher);

      unstrippedDeployArchiveBuilder.build();
    } else {
      // Write an empty file as the name_deploy.jar.unstripped when the default output jar is not
      // stripped.
      ruleContext.registerAction(
          FileWriteAction.create(ruleContext, unstrippedDeployJar, "", false));
    }

    JavaRuleOutputJarsProvider ruleOutputJarsProvider = ruleOutputJarsProviderBuilder.build();

    JavaInfo.Builder javaInfoBuilder = JavaInfo.Builder.create();

    NestedSetBuilder<Pair<String, String>> coverageEnvironment = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Artifact> coverageSupportFiles = NestedSetBuilder.stableOrder();
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {

      // Create an artifact that contains the root relative paths of the jars on the runtime
      // classpath.
      Artifact runtimeClasspathArtifact =
          ruleContext.getUniqueDirectoryArtifact(
              "runtime_classpath_for_coverage",
              "runtime_classpath.txt",
              ruleContext.getBinOrGenfilesDirectory());
      ruleContext.registerAction(
          new LazyWritePathsFileAction(
              ruleContext.getActionOwner(),
              runtimeClasspathArtifact,
              common.getRuntimeClasspath(),
              /* filesToIgnore= */ ImmutableSet.of(),
              true));
      filesBuilder.add(runtimeClasspathArtifact);

      // Pass the artifact through an environment variable in the coverage environment so it
      // can be read by the coverage collection script.
      coverageEnvironment.add(
          new Pair<>(
              "JAVA_RUNTIME_CLASSPATH_FOR_COVERAGE", runtimeClasspathArtifact.getExecPathString()));
      // Add the file to coverageSupportFiles so it ends up as an input for the test action
      // when coverage is enabled.
      coverageSupportFiles.add(runtimeClasspathArtifact);

      // Make single jar reachable from the coverage environment because it needs to be executed
      // by the coverage collection script.
      Artifact singleJar = JavaToolchainProvider.from(ruleContext).getSingleJar();
      coverageEnvironment.add(new Pair<>("SINGLE_JAR_TOOL", singleJar.getExecPathString()));
      coverageSupportFiles.add(singleJar);
      runfilesBuilder.addArtifact(singleJar);
      runfilesBuilder.addArtifact(runtimeClasspathArtifact);
    }

    common.addTransitiveInfoProviders(
        builder,
        javaInfoBuilder,
        filesToBuild,
        classJar,
        coverageEnvironment.build(),
        coverageSupportFiles.build());
    common.addGenJarsProvider(builder, javaInfoBuilder, outputs.genClass(), outputs.genSource());

    JavaInfo javaInfo =
        javaInfoBuilder
            .addProvider(JavaSourceJarsProvider.class, sourceJarsProvider)
            .addProvider(JavaRuleOutputJarsProvider.class, ruleOutputJarsProvider)
            .addProvider(
                JavaSourceInfoProvider.class,
                JavaSourceInfoProvider.fromJavaTargetAttributes(attributes, semantics))
            .build();

    return builder
        .setFilesToBuild(filesToBuild)
        .addNativeDeclaredProvider(javaInfo)
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
        .add(RunfilesProvider.class, runfilesProvider)
        // The executable to run (below) may be different from the executable for runfiles (the one
        // we create the runfiles support object with). On Linux they are the same (it's the same
        // shell script), on Windows they are different (the executable to run is a batch file, the
        // executable for runfiles is the shell script).
        .setRunfilesSupport(runfilesSupport, executableToRun)
        .setPersistentTestRunnerRunfiles(persistentTestRunnerRunfiles)
        // Add the native libraries as test action tools. Useful for the persistent test runner
        // to include them in the worker's key and re-build a worker if the native dependencies
        // have changed.
        .addTestActionTools(nativeLibraries)
        .addFilesToRun(extraFilesToRunBuilder.build())
        .add(
            JavaRuntimeClasspathProvider.class,
            new JavaRuntimeClasspathProvider(common.getRuntimeClasspath()))
        .addOutputGroup(JavaSemantics.SOURCE_JARS_OUTPUT_GROUP, transitiveSourceJars)
        .build();
  }

  // Create the deploy jar and make it dependent on the runfiles middleman if an executable is
  // created. Do not add the deploy jar to files to build, so we will only build it when it gets
  // requested.
  private ImmutableList<String> getDeployManifestLines(
      RuleContext ruleContext, String originalMainClass) {
    ImmutableList.Builder<String> builder =
        ImmutableList.<String>builder()
            .addAll(ruleContext.attributes().get("deploy_manifest_lines", Type.STRING_LIST));
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      builder.add("Coverage-Main-Class: " + originalMainClass);
    }
    return builder.build();
  }

  /** Add Java8 timezone resource jar to java binary, if specified in tool chain. */
  private void addTimezoneResourceForJavaBinaries(
      RuleContext ruleContext, JavaTargetAttributes.Builder attributesBuilder) {
    JavaToolchainProvider toolchainProvider = JavaToolchainProvider.from(ruleContext);
    if (toolchainProvider.getTimezoneData() != null) {
      attributesBuilder.addResourceJars(
          NestedSetBuilder.create(Order.STABLE_ORDER, toolchainProvider.getTimezoneData()));
    }
  }

  private void collectDefaultRunfiles(
      Runfiles.Builder builder,
      RuleContext ruleContext,
      JavaCommon common,
      JavaCompilationArtifacts javaArtifacts,
      NestedSet<Artifact> filesToBuild,
      Artifact launcher,
      NestedSet<Artifact> dynamicRuntimeActionInputs) {
    builder.addTransitiveArtifactsWrappedInStableOrder(filesToBuild);
    builder.addArtifacts(javaArtifacts.getRuntimeJars());
    if (launcher != null) {
      final TransitiveInfoCollection defaultLauncher =
          JavaHelper.launcherForTarget(semantics, ruleContext);
      final Artifact defaultLauncherArtifact =
          JavaHelper.launcherArtifactForTarget(semantics, ruleContext);
      if (!defaultLauncherArtifact.equals(launcher)) {
        builder.addArtifact(launcher);

        // N.B. The "default launcher" referred to here is the launcher target specified through
        // an attribute or flag. We wish to retain the runfiles of the default launcher, *except*
        // for the original cc_binary artifact, because we've swapped it out with our custom
        // launcher. Hence, instead of calling builder.addTarget(), or adding an odd method
        // to Runfiles.Builder, we "unravel" the call and manually add things to the builder.
        // Because the NestedSet representing each target's launcher runfiles is re-built here,
        // we may see increased memory consumption for representing the target's runfiles.
        Runfiles runfiles =
            defaultLauncher.getProvider(RunfilesProvider.class).getDefaultRunfiles();
        NestedSetBuilder<Artifact> unconditionalArtifacts = NestedSetBuilder.compileOrder();
        for (Artifact a : runfiles.getUnconditionalArtifacts().toList()) {
          if (!a.equals(defaultLauncherArtifact)) {
            unconditionalArtifacts.add(a);
          }
        }
        builder.addTransitiveArtifacts(unconditionalArtifacts.build());
        builder.addSymlinks(runfiles.getSymlinks());
        builder.addRootSymlinks(runfiles.getRootSymlinks());
        builder.addPruningManifests(runfiles.getPruningManifests());
      } else {
        builder.addTarget(defaultLauncher, RunfilesProvider.DEFAULT_RUNFILES);
      }
    }

    semantics.addRunfilesForBinary(ruleContext, launcher, builder);
    builder.addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES);
    builder.add(ruleContext, JavaRunfilesProvider.TO_RUNFILES);

    List<? extends TransitiveInfoCollection> runtimeDeps =
        ruleContext.getPrerequisites("runtime_deps", Mode.TARGET);
    builder.addTargets(runtimeDeps, JavaRunfilesProvider.TO_RUNFILES);
    builder.addTargets(runtimeDeps, RunfilesProvider.DEFAULT_RUNFILES);

    builder.addTransitiveArtifactsWrappedInStableOrder(common.getRuntimeClasspath());

    // Add the JDK files if it comes from the source repository (see java_stub_template.txt).
    JavaRuntimeInfo javaRuntime = JavaRuntimeInfo.from(ruleContext);
    if (javaRuntime != null) {
      builder.addTransitiveArtifacts(javaRuntime.javaBaseInputs());

      if (!javaRuntime.javaHomePathFragment().isAbsolute()) {
        // Add symlinks to the C++ runtime libraries under a path that can be built
        // into the Java binary without having to embed the crosstool, gcc, and grte
        // version information contained within the libraries' package paths.
        for (Artifact lib : dynamicRuntimeActionInputs.toList()) {
          PathFragment path = CPP_RUNTIMES.getRelative(lib.getExecPath().getBaseName());
          builder.addSymlink(path, lib);
        }
      }
    }
  }

  /**
   * Collects the native libraries in the transitive closure of the deps.
   *
   * @param deps the dependencies to be included as roots of the transitive closure.
   * @return the native libraries found in the transitive closure of the deps.
   */
  public static Collection<Artifact> collectNativeLibraries(
      Iterable<? extends TransitiveInfoCollection> deps) {
    NestedSet<LibraryToLink> linkerInputs =
        new NativeLibraryNestedSetBuilder().addJavaTargets(deps).build();
    return LibraryToLink.getDynamicLibrariesForLinking(linkerInputs);
  }

  private static boolean isJavaTestRule(RuleContext ruleContext) {
    return ruleContext.getRule().getRuleClass().endsWith("_test");
  }
}
