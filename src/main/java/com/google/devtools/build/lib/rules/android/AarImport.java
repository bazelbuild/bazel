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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.rules.java.JavaCompilationArtifacts;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.JavaSkylarkApiProvider;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.vfs.PathFragment;

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

  protected AarImport(JavaSemantics javaSemantics) {
    this.javaSemantics = javaSemantics;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
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
    ruleContext.registerAction(
        createSingleFileExtractorActions(
            ruleContext, aar, ANDROID_MANIFEST, androidManifestArtifact));

    Artifact resources = createAarTreeArtifact(ruleContext, "resources");
    ruleContext.registerAction(createAarResourcesExtractorActions(ruleContext, aar, resources));

    ApplicationManifest androidManifest =
        ApplicationManifest.fromExplicitManifest(ruleContext, androidManifestArtifact);

    FileProvider resourcesProvider =
        new FileProvider(
            new NestedSetBuilder<Artifact>(Order.NAIVE_LINK_ORDER).add(resources).build());

    Artifact resourcesZip =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP);

    ResourceApk resourceApk =
        androidManifest.packAarWithDataAndResources(
            ruleContext,
            LocalResourceContainer.forResourceFileProvider(
                ruleContext, resourcesProvider, "resources"),
            ResourceDependencies.fromRuleDeps(ruleContext, JavaCommon.isNeverLink(ruleContext)),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_LOCAL_SYMBOLS),
            ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
            resourcesZip);

    // There isn't really any use case for building an aar_import target on its own, so the files to
    // build could be empty. The resources zip and merged jars are added here as a sanity check for
    // Bazel developers so that `bazel build java/com/my_aar_import` will fail if the resource
    // processing or jar merging steps fail.
    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().add(resourcesZip).add(mergedJar);

    Artifact nativeLibs = createAarArtifact(ruleContext, "native_libs.zip");
    ruleContext.registerAction(createAarNativeLibsFilterActions(ruleContext, aar, nativeLibs));

    JavaRuleOutputJarsProvider.Builder jarProviderBuilder =
        new JavaRuleOutputJarsProvider.Builder().addOutputJar(mergedJar, null, ImmutableList.of());
    for (TransitiveInfoCollection export : ruleContext.getPrerequisites("exports", Mode.TARGET)) {
      for (OutputJar jar :
          JavaInfo.getProvider(JavaRuleOutputJarsProvider.class, export).getOutputJars()) {
        jarProviderBuilder.addOutputJar(jar);
        filesToBuildBuilder.add(jar.getClassJar());
      }
    }

    ImmutableList<TransitiveInfoCollection> targets =
        ImmutableList.copyOf(ruleContext.getPrerequisites("exports", Mode.TARGET));
    JavaCommon common =
        new JavaCommon(
            ruleContext,
            javaSemantics,
            /* sources = */ ImmutableList.of(),
            /* compileDeps = */ targets,
            /* runtimeDeps = */ targets,
            /* bothDeps = */ targets);
    common.setJavaCompilationArtifacts(
        new JavaCompilationArtifacts.Builder()
            .addRuntimeJar(mergedJar)
            .addCompileTimeJarAsFullJar(mergedJar)
            .build());

    JavaInfo javaInfo =
        JavaInfo.Builder.create()
            .addProvider(
                JavaCompilationArgsProvider.class,
                JavaCompilationArgsProvider.create(
                    common.collectJavaCompilationArgs(
                        /* recursive = */ false,
                        JavaCommon.isNeverLink(ruleContext),
                        /* srcLessDepsExport = */ false),
                    common.collectJavaCompilationArgs(
                        /* recursive = */ true,
                        JavaCommon.isNeverLink(ruleContext),
                        /* srcLessDepsExport = */ false)))
            .addProvider(JavaRuleOutputJarsProvider.class, jarProviderBuilder.build())
            .build();

    return ruleBuilder
        .setFilesToBuild(filesToBuildBuilder.build())
        .addSkylarkTransitiveInfo(
            JavaSkylarkApiProvider.NAME, JavaSkylarkApiProvider.fromRuleContext())
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(
            AndroidResourcesProvider.class,
            resourceApk.toResourceProvider(ruleContext.getLabel()))
        .addProvider(
            NativeLibsZipsProvider.class,
            new NativeLibsZipsProvider(
                AndroidCommon.collectTransitiveNativeLibsZips(ruleContext).add(nativeLibs).build()))
        .addProvider(
            JavaRuntimeJarProvider.class, new JavaRuntimeJarProvider(ImmutableList.of(mergedJar)))
        .addNativeDeclaredProvider(javaInfo)
        .build();
  }

  private static Action[] createSingleFileExtractorActions(
      RuleContext ruleContext, Artifact aar, String filename, Artifact outputArtifact) {
    return new SpawnAction.Builder()
        .useDefaultShellEnvironment()
        .setExecutable(ruleContext.getExecutablePrerequisite(AarImportBaseRule.ZIPPER, Mode.HOST))
        .setMnemonic("AarFileExtractor")
        .setProgressMessage("Extracting %s from %s", filename, aar.getFilename())
        .addInput(aar)
        .addOutput(outputArtifact)
        .addCommandLine(
            CustomCommandLine.builder()
                .addExecPath("x", aar)
                .addPath("-d", outputArtifact.getExecPath().getParentDirectory())
                .addDynamicString(filename)
                .build())
        .build(ruleContext);
  }

  private static Action[] createAarResourcesExtractorActions(
      RuleContext ruleContext, Artifact aar, Artifact outputTree) {
    return new SpawnAction.Builder()
        .useDefaultShellEnvironment()
        .setExecutable(
            ruleContext.getExecutablePrerequisite(
                AarImportBaseRule.AAR_RESOURCES_EXTRACTOR, Mode.HOST))
        .setMnemonic("AarResourcesExtractor")
        .addInput(aar)
        .addOutput(outputTree)
        .addCommandLine(
            CustomCommandLine.builder()
                .addExecPath("--input_aar", aar)
                .addExecPath("--output_res_dir", outputTree)
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

  private static Artifact createAarTreeArtifact(RuleContext ruleContext, String name) {
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
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext));
    } else {
      builder.setExecutable(singleJar);
    }
    return builder;
  }
}
