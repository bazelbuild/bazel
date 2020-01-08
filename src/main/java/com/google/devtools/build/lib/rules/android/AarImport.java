// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.databinding.DataBinding;
import com.google.devtools.build.lib.rules.java.ImportDepsCheckActionBuilder;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration.ImportDepsCheckingLevel;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuntimeInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceInfoProvider;
import com.google.devtools.build.lib.rules.java.JavaSourceJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * An implementation for the aar_import rule.
 *
 * <p>AAR files are zip archives that contain an Android Manifest, JARs, resources, assets, native
 * libraries, Proguard configuration and lint jars. Currently the aar_import rule supports AARs with
 * an AndroidManifest.xml, classes.jar, libs/, res/ and jni/. Assets are not yet supported.
 *
 * @see <a href="http://tools.android.com/tech-docs/new-build-system/aar-format">AAR Format</a>
 */
public class AarImport implements RuleConfiguredTargetFactory {
  private static final String ANDROID_MANIFEST = "AndroidManifest.xml";
  private static final String MERGED_JAR = "classes_and_libs_merged.jar";

  private final JavaSemantics javaSemantics;
  private final AndroidSemantics androidSemantics;

  protected AarImport(JavaSemantics javaSemantics, AndroidSemantics androidSemantics) {
    this.javaSemantics = javaSemantics;
    this.androidSemantics = androidSemantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    androidSemantics.checkForMigrationTag(ruleContext);
    AndroidSdkProvider.verifyPresence(ruleContext);

    RuleConfiguredTargetBuilder ruleBuilder = new RuleConfiguredTargetBuilder(ruleContext);
    Artifact aar = ruleContext.getPrerequisiteArtifact("aar", Mode.TARGET);

    Artifact allAarJars = createAarTreeArtifact(ruleContext, "jars");
    Artifact jarMergingParams = createAarArtifact(ruleContext, "jar_merging_params");
    ruleContext.registerAction(
        createAarEmbeddedJarsExtractorActions(ruleContext, aar, allAarJars, jarMergingParams));
    Artifact mergedJar = createAarArtifact(ruleContext, MERGED_JAR);
    ruleContext.registerAction(
        createAarJarsMergingActions(ruleContext, allAarJars, mergedJar, jarMergingParams));

    // AndroidManifest.xml is required in every AAR.
    Artifact androidManifestArtifact = createAarArtifact(ruleContext, ANDROID_MANIFEST);

    SpecialArtifact resources = createAarTreeArtifact(ruleContext, "resources");
    SpecialArtifact assets = createAarTreeArtifact(ruleContext, "assets");
    ruleContext.registerAction(
        createAarResourcesExtractorActions(ruleContext, aar, resources, assets));

    AndroidDataContext dataContext = androidSemantics.makeContextForNative(ruleContext);
    StampedAndroidManifest manifest = AndroidManifest.forAarImport(androidManifestArtifact);

    boolean neverlink = JavaCommon.isNeverLink(ruleContext);

    ValidatedAndroidResources validatedResources =
        AndroidResources.forAarImport(resources)
            .process(
                ruleContext,
                dataContext,
                manifest,
                DataBinding.contextFrom(ruleContext, dataContext.getAndroidConfig()),
                neverlink);

    MergedAndroidAssets mergedAssets =
        AndroidAssets.forAarImport(assets)
            .process(dataContext, AssetDependencies.fromRuleDeps(ruleContext, neverlink));

    ResourceApk resourceApk = ResourceApk.of(validatedResources, mergedAssets, null, null);

    // There isn't really any use case for building an aar_import target on its own, so the files to
    // build could be empty. The R class JAR and merged JARs are added here as a sanity check for
    // Bazel developers so that `bazel build java/com/my_aar_import` will fail if the resource
    // processing or JAR merging steps fail.
    NestedSet<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(resourceApk.getValidatedResources().getJavaClassJar())
            .add(mergedJar)
            .build();

    Artifact nativeLibs = createAarArtifact(ruleContext, "native_libs.zip");
    ruleContext.registerAction(createAarNativeLibsFilterActions(ruleContext, aar, nativeLibs));

    JavaRuleOutputJarsProvider.Builder jarProviderBuilder =
        new JavaRuleOutputJarsProvider.Builder()
            .addOutputJar(mergedJar, null /* ijar */, null /* manifestProto */, ImmutableList.of());

    ImmutableList<TransitiveInfoCollection> targets =
        ImmutableList.<TransitiveInfoCollection>builder()
            .addAll(ruleContext.getPrerequisites("exports", Mode.TARGET))
            .addAll(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .build();
    JavaCommon common =
        new JavaCommon(
            ruleContext,
            javaSemantics,
            /* sources = */ ImmutableList.of(),
            /* compileDeps = */ targets,
            /* runtimeDeps = */ targets,
            /* bothDeps = */ targets);
    javaSemantics.checkRule(ruleContext, common);

    JavaConfiguration javaConfig = ruleContext.getFragment(JavaConfiguration.class);
    JavaCompilationArtifacts.Builder javaCompilationArtifactsBuilder =
        new JavaCompilationArtifacts.Builder();

    javaCompilationArtifactsBuilder
        .addRuntimeJar(mergedJar)
        .addCompileTimeJarAsFullJar(mergedJar)
        // Allow direct dependents to compile against un-merged R classes
        .addCompileTimeJarAsFullJar(
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_CLASS_JAR));

    Artifact jdepsArtifact = null;
    // Don't register import deps checking actions if the level is off. Since it's off, the
    // check isn't useful anyway, so don't waste resources running it.
    if (javaConfig.getImportDepsCheckingLevel() != ImportDepsCheckingLevel.OFF) {
      jdepsArtifact = createAarArtifact(ruleContext, "jdeps.proto");
      javaCompilationArtifactsBuilder.setCompileTimeDependencies(jdepsArtifact);
      ImportDepsCheckActionBuilder.newBuilder()
          .bootclasspath(getBootclasspath(ruleContext))
          .declareDeps(getCompileTimeJarsFromCollection(targets, /*isDirect=*/ true))
          .transitiveDeps(getCompileTimeJarsFromCollection(targets, /*isDirect=*/ false))
          .checkJars(NestedSetBuilder.<Artifact>stableOrder().add(mergedJar).build())
          .importDepsCheckingLevel(javaConfig.getImportDepsCheckingLevel())
          .jdepsOutputArtifact(jdepsArtifact)
          .ruleLabel(ruleContext.getLabel())
          .buildAndRegister(ruleContext);
    }

    common.setJavaCompilationArtifacts(javaCompilationArtifactsBuilder.build());

    // We pass jdepsArtifact to create the action of extracting ANDROID_MANIFEST. Note that
    // this action does not need jdepsArtifact. The only reason is that we need to check the
    // dependencies of this aar_import, and we need to put its result on the build graph so that the
    // dependency checking action is called.
    ruleContext.registerAction(
        createSingleFileExtractorActions(
            ruleContext, aar, ANDROID_MANIFEST, jdepsArtifact, androidManifestArtifact));

    JavaCompilationArgsProvider javaCompilationArgsProvider =
        common.collectJavaCompilationArgs(
            /* isNeverLink = */ JavaCommon.isNeverLink(ruleContext),
            /* srcLessDepsExport = */ false);

    // Wire up the source jar for the current target and transitive source jars from dependencies.
    ImmutableList<Artifact> srcJars = ImmutableList.of();
    Artifact srcJar = ruleContext.getPrerequisiteArtifact("srcjar", Mode.TARGET);
    NestedSetBuilder<Artifact> transitiveJavaSourceJarBuilder = NestedSetBuilder.stableOrder();
    if (srcJar != null) {
      srcJars = ImmutableList.of(srcJar);
      transitiveJavaSourceJarBuilder.add(srcJar);
    }
    for (JavaSourceJarsProvider other :
        JavaInfo.getProvidersFromListOfTargets(
            JavaSourceJarsProvider.class, ruleContext.getPrerequisites("exports", Mode.TARGET))) {
      transitiveJavaSourceJarBuilder.addTransitive(other.getTransitiveSourceJars());
    }
    NestedSet<Artifact> transitiveJavaSourceJars = transitiveJavaSourceJarBuilder.build();
    JavaSourceJarsProvider javaSourceJarsProvider =
        JavaSourceJarsProvider.create(transitiveJavaSourceJars, srcJars);
    JavaSourceInfoProvider javaSourceInfoProvider =
        new JavaSourceInfoProvider.Builder()
            .setJarFiles(ImmutableList.of(mergedJar))
            .setSourceJarsForJarFiles(srcJars)
            .build();

    JavaInfo.Builder javaInfoBuilder =
        JavaInfo.Builder.create()
            .setRuntimeJars(ImmutableList.of(mergedJar))
            .setJavaConstraints(ImmutableList.of("android"))
            .setNeverlink(JavaCommon.isNeverLink(ruleContext))
            .addProvider(JavaCompilationArgsProvider.class, javaCompilationArgsProvider)
            .addProvider(JavaSourceJarsProvider.class, javaSourceJarsProvider)
            .addProvider(JavaSourceInfoProvider.class, javaSourceInfoProvider)
            .addProvider(JavaRuleOutputJarsProvider.class, jarProviderBuilder.build());

    common.addTransitiveInfoProviders(
        ruleBuilder, javaInfoBuilder, filesToBuild, /*classJar=*/ null);

    resourceApk.addToConfiguredTargetBuilder(
        ruleBuilder,
        ruleContext.getLabel(),
        /* includeSkylarkApiProvider = */ false,
        /* isLibrary = */ true);

    ruleBuilder
        .setFilesToBuild(filesToBuild)
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addNativeDeclaredProvider(
            new AndroidNativeLibsInfo(
                AndroidCommon.collectTransitiveNativeLibs(ruleContext).add(nativeLibs).build()))
        .addNativeDeclaredProvider(javaInfoBuilder.build());
    if (jdepsArtifact != null) {
      // Add the deps check result so that we can unit test it.
      ruleBuilder.addOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL, jdepsArtifact);
    }
    return ruleBuilder.build();
  }

  private static NestedSet<Artifact> getCompileTimeJarsFromCollection(
      ImmutableList<TransitiveInfoCollection> deps, boolean isDirect) {
    JavaCompilationArgsProvider provider = JavaCompilationArgsProvider.legacyFromTargets(deps);
    return isDirect ? provider.getDirectCompileTimeJars() : provider.getTransitiveCompileTimeJars();
  }

  private NestedSet<Artifact> getBootclasspath(RuleContext ruleContext) {
    if (AndroidCommon.getAndroidConfig(ruleContext).desugarJava8()) {
      return NestedSetBuilder.<Artifact>stableOrder()
          .addTransitive(
              ruleContext
                  .getPrerequisite("$desugar_java8_extra_bootclasspath", Mode.HOST)
                  .getProvider(FileProvider.class)
                  .getFilesToBuild())
          .add(AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar())
          .build();
    } else {
      return NestedSetBuilder.<Artifact>stableOrder()
          .add(AndroidSdkProvider.fromRuleContext(ruleContext).getAndroidJar())
          .build();
    }
  }

  /**
   * Create an action to extract a file (specified by the parameter filename) from an AAR file. Note
   * that the parameter jdepsOutputArtifact is not necessary for this action. Conversely, the action
   * of checking dependencies for aar_import needs this action instead. Therefore we add the output
   * artifact of import_deps_checker to this extraction action as input. Therefore, the dependency
   * checking will run each time.
   */
  private static Action[] createSingleFileExtractorActions(
      RuleContext ruleContext,
      Artifact aar,
      String filename,
      @Nullable Artifact jdepsOutputArtifact,
      Artifact outputArtifact) {
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(
                ruleContext.getExecutablePrerequisite(AarImportBaseRule.ZIPPER, Mode.HOST))
            .setMnemonic("AarFileExtractor")
            .setProgressMessage("Extracting %s from %s", filename, aar.getFilename())
            .addInput(aar)
            .addOutput(outputArtifact)
            .addCommandLine(
                CustomCommandLine.builder()
                    .addExecPath("x", aar)
                    .addPath("-d", outputArtifact.getExecPath().getParentDirectory())
                    .addDynamicString(filename)
                    .build());
    if (jdepsOutputArtifact != null) {
      builder.addInput(jdepsOutputArtifact);
    }
    return builder.build(ruleContext);
  }

  private static Action[] createAarResourcesExtractorActions(
      RuleContext ruleContext, Artifact aar, Artifact resourcesDir, Artifact assetsDir) {
    return new SpawnAction.Builder()
        .useDefaultShellEnvironment()
        .setExecutable(
            ruleContext.getExecutablePrerequisite(
                AarImportBaseRule.AAR_RESOURCES_EXTRACTOR, Mode.HOST))
        .setMnemonic("AarResourcesExtractor")
        .addInput(aar)
        .addOutput(resourcesDir)
        .addOutput(assetsDir)
        .addCommandLine(
            CustomCommandLine.builder()
                .addExecPath("--input_aar", aar)
                .addExecPath("--output_res_dir", resourcesDir)
                .addExecPath("--output_assets_dir", assetsDir)
                .build())
        .build(ruleContext);
  }

  private static Action[] createAarEmbeddedJarsExtractorActions(
      RuleContext ruleContext,
      Artifact aar,
      Artifact jarsTreeArtifact,
      Artifact singleJarParamFile) {
    return new SpawnAction.Builder()
        .useDefaultShellEnvironment()
        .setExecutable(
            ruleContext.getExecutablePrerequisite(
                AarImportBaseRule.AAR_EMBEDDED_JARS_EXTACTOR, Mode.HOST))
        .setMnemonic("AarEmbeddedJarsExtractor")
        .setProgressMessage("Extracting classes.jar and libs/*.jar from %s", aar.getFilename())
        .addInput(aar)
        .addOutput(jarsTreeArtifact)
        .addOutput(singleJarParamFile)
        .addCommandLine(
            CustomCommandLine.builder()
                .addExecPath("--input_aar", aar)
                .addExecPath("--output_dir", jarsTreeArtifact)
                .addExecPath("--output_singlejar_param_file", singleJarParamFile)
                .build())
        .build(ruleContext);
  }

  private static Action[] createAarJarsMergingActions(
      RuleContext ruleContext, Artifact jarsTreeArtifact, Artifact mergedJar, Artifact paramFile) {
    return singleJarSpawnActionBuilder(ruleContext)
        .setMnemonic("AarJarsMerger")
        .setProgressMessage("Merging AAR embedded jars")
        .addInput(jarsTreeArtifact)
        .addOutput(mergedJar)
        .addInput(paramFile)
        .addCommandLine(
            CustomCommandLine.builder()
                .addExecPath("--output", mergedJar)
                .add("--dont_change_compression")
                .add("--normalize")
                .addPrefixedExecPath("@", paramFile)
                .build())
        .build(ruleContext);
  }

  private static Action[] createAarNativeLibsFilterActions(
      RuleContext ruleContext, Artifact aar, Artifact outputZip) {
    SpawnAction.Builder actionBuilder =
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .setExecutable(
                ruleContext.getExecutablePrerequisite(
                    AarImportBaseRule.AAR_NATIVE_LIBS_ZIP_CREATOR, Mode.HOST))
            .setMnemonic("AarNativeLibsFilter")
            .setProgressMessage("Filtering AAR native libs by architecture")
            .addInput(aar)
            .addOutput(outputZip)
            .addCommandLine(
                CustomCommandLine.builder()
                    .addExecPath("--input_aar", aar)
                    .add("--cpu", ruleContext.getConfiguration().getCpu())
                    .addExecPath("--output_zip", outputZip)
                    .build());
    return actionBuilder.build(ruleContext);
  }

  private static Artifact createAarArtifact(RuleContext ruleContext, String name) {
    return ruleContext.getUniqueDirectoryArtifact(
        "_aar", name, ruleContext.getBinOrGenfilesDirectory());
  }

  private static SpecialArtifact createAarTreeArtifact(RuleContext ruleContext, String name) {
    PathFragment rootRelativePath = ruleContext.getUniqueDirectory("_aar/unzipped/" + name);
    return ruleContext.getTreeArtifact(rootRelativePath, ruleContext.getBinOrGenfilesDirectory());
  }

  // Adds the appropriate SpawnAction options depending on if SingleJar is a jar or not.
  private static SpawnAction.Builder singleJarSpawnActionBuilder(RuleContext ruleContext) {
    SpawnAction.Builder builder = new SpawnAction.Builder().useDefaultShellEnvironment();
    Artifact singleJar = JavaToolchainProvider.from(ruleContext).getSingleJar();
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .setJarExecutable(
              JavaCommon.getHostJavaExecutable(ruleContext),
              singleJar,
              JavaToolchainProvider.from(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaRuntimeInfo.forHost(ruleContext).javaBaseInputsMiddleman());
    } else {
      builder.setExecutable(singleJar);
    }
    return builder;
  }
}
