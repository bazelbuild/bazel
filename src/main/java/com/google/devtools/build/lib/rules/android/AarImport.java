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
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaHelper;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.rules.java.JavaRuntimeJarProvider;
import com.google.devtools.build.lib.rules.java.JavaToolchainProvider;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * An implementation for the aar_import rule.
 *
 * AAR files are zip archives that contain an Android Manifest, JARs, resources, assets, native
 * libraries, Proguard configuration and lint jars. Currently the aar_import rule supports AARs with
 * an AndroidManifest.xml, classes.jar, libs/, res/ and jni/. Assets are not yet supported.
 *
 * @see <a href="http://tools.android.com/tech-docs/new-build-system/aar-format">AAR Format</a>
 */
public class AarImport implements RuleConfiguredTargetFactory {
  private static final String ANDROID_MANIFEST = "AndroidManifest.xml";
  private static final String MERGED_JAR = "classes_and_libs_merged.jar";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
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
    ruleContext.registerAction(createSingleFileExtractorActions(
        ruleContext, aar, ANDROID_MANIFEST, androidManifestArtifact));

    Artifact resourcesManifest = createAarArtifact(ruleContext, "resource_manifest");
    ruleContext.registerAction(
        createManifestExtractorActions(ruleContext, aar, "res/.*", resourcesManifest));

    Artifact resources = createAarTreeArtifact(ruleContext, "resources");
    ruleContext.registerAction(
        createManifestFileEntriesExtractorActions(ruleContext, aar, resourcesManifest, resources));

    ApplicationManifest androidManifest =
        ApplicationManifest.fromExplicitManifest(ruleContext, androidManifestArtifact);

    FileProvider resourcesProvider = new FileProvider(
        new NestedSetBuilder<Artifact>(Order.NAIVE_LINK_ORDER).add(resources).build());

    Artifact resourcesZip =
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP);

    ResourceApk resourceApk = androidManifest.packWithDataAndResources(
        ruleContext,
        new LocalResourceContainer.Builder(ruleContext)
            .withResources(ImmutableList.of(resourcesProvider))
            .build(),
        ResourceDependencies.fromRuleDeps(ruleContext, JavaCommon.isNeverLink(ruleContext)),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_SYMBOLS_TXT),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
        resourcesZip,
        /* alwaysExportManifest = */ true);

    // There isn't really any use case for building an aar_import target on its own, so the files to
    // build could be empty. The resources zip and merged jars are added here as a sanity check for
    // Bazel developers so that `bazel build java/com/my_aar_import` will fail if the resource
    // processing or jar merging steps fail.
    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().add(resourcesZip).add(mergedJar);

    Artifact nativeLibs = createAarArtifact(ruleContext, "native_libs.zip");
    ruleContext.registerAction(createAarNativeLibsFilterActions(ruleContext, aar, nativeLibs));

    JavaRuleOutputJarsProvider.Builder jarProviderBuilder = new JavaRuleOutputJarsProvider.Builder()
        .addOutputJar(mergedJar, null, null);
    for (TransitiveInfoCollection export : ruleContext.getPrerequisites("exports", Mode.TARGET)) {
      for (OutputJar jar : export.getProvider(JavaRuleOutputJarsProvider.class).getOutputJars()) {
        jarProviderBuilder.addOutputJar(jar);
        filesToBuildBuilder.add(jar.getClassJar());
      }
    }

    return ruleBuilder
        .setFilesToBuild(filesToBuildBuilder.build())
        .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .addProvider(
            AndroidResourcesProvider.class, resourceApk.toResourceProvider(ruleContext.getLabel()))
        .addProvider(
            NativeLibsZipsProvider.class,
            new NativeLibsZipsProvider(
                AndroidCommon.collectTransitiveNativeLibsZips(ruleContext).add(nativeLibs).build()))
        .addProvider(
            JavaRuntimeJarProvider.class, new JavaRuntimeJarProvider(ImmutableList.of(mergedJar)))
        .addProvider(JavaRuleOutputJarsProvider.class, jarProviderBuilder.build())
        .build();
  }

  private static Action[] createSingleFileExtractorActions(RuleContext ruleContext, Artifact aar,
      String filename, Artifact outputArtifact) {
    return new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$zipper", Mode.HOST))
        .setMnemonic("AarFileExtractor")
        .setProgressMessage("Extracting " + filename + " from " + aar.getFilename())
        .addArgument("x")
        .addInputArgument(aar)
        .addArgument("-d")
        .addOutput(outputArtifact)
        .addArgument(outputArtifact.getExecPath().getParentDirectory().getPathString())
        .addArgument(filename)
        .build(ruleContext);
  }

  private static Action[] createManifestFileEntriesExtractorActions(RuleContext ruleContext,
      Artifact aar, Artifact manifest, Artifact outputTree) {
    return new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$zipper", Mode.HOST))
        .setMnemonic("AarManifestFileEntriesExtractor")
        .addArgument("x")
        .addInputArgument(aar)
        .addArgument("-d")
        .addOutputArgument(outputTree)
        .addArgument("@" + manifest.getExecPathString())
        .addInput(manifest)
        .build(ruleContext);
  }

  private static Action[] createAarEmbeddedJarsExtractorActions(RuleContext ruleContext,
      Artifact aar, Artifact jarsTreeArtifact, Artifact singleJarParamFile) {
    return new SpawnAction.Builder()
        .setExecutable(
            ruleContext.getExecutablePrerequisite("$aar_embedded_jars_extractor", Mode.HOST))
        .setMnemonic("AarEmbeddedJarsExtractor")
        .setProgressMessage("Extracting classes.jar and libs/*.jar from " + aar.getFilename())
        .addArgument("--input_aar")
        .addInputArgument(aar)
        .addArgument("--output_dir")
        .addOutputArgument(jarsTreeArtifact)
        .addArgument("--output_singlejar_param_file")
        .addOutputArgument(singleJarParamFile)
        .build(ruleContext);
  }

  private static Action[] createAarJarsMergingActions(RuleContext ruleContext,
      Artifact jarsTreeArtifact, Artifact mergedJar, Artifact paramFile) {
    return singleJarSpawnActionBuilder(ruleContext)
        .setMnemonic("AarJarsMerger")
        .setProgressMessage("Merging AAR embedded jars")
        .addInput(jarsTreeArtifact)
        .addArgument("--output")
        .addOutputArgument(mergedJar)
        .addArgument("--dont_change_compression")
        .addInput(paramFile)
        .addArgument("@" + paramFile.getExecPathString())
        .build(ruleContext);
  }

  private static Action[] createManifestExtractorActions(RuleContext ruleContext, Artifact aar,
      String filenameRegexp, Artifact manifest) {
    return new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$zip_manifest_creator", Mode.HOST))
        .setMnemonic("ZipManifestCreator")
        .setProgressMessage(
            "Creating manifest for " + aar.getFilename() + " matching " + filenameRegexp)
        .addArgument(filenameRegexp)
        .addInputArgument(aar)
        .addOutputArgument(manifest)
        .build(ruleContext);
  }

  private static Action[] createAarNativeLibsFilterActions(RuleContext ruleContext, Artifact aar,
      Artifact outputZip) {
    SpawnAction.Builder actionBuilder = new SpawnAction.Builder()
        .setExecutable(
            ruleContext.getExecutablePrerequisite("$aar_native_libs_zip_creator", Mode.HOST))
        .setMnemonic("AarNativeLibsFilter")
        .setProgressMessage("Filtering AAR native libs by architecture")
        .addArgument("--input_aar")
        .addInputArgument(aar)
        .addArgument("--cpu")
        .addArgument(ruleContext.getConfiguration().getCpu())
        .addArgument("--output_zip")
        .addOutputArgument(outputZip);
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
    SpawnAction.Builder builder = new SpawnAction.Builder();
    Artifact singleJar = JavaToolchainProvider.fromRuleContext(ruleContext).getSingleJar();
    if (singleJar.getFilename().endsWith(".jar")) {
      builder
          .setJarExecutable(
              ruleContext.getHostConfiguration().getFragment(Jvm.class).getJavaExecutable(),
              singleJar,
              JavaToolchainProvider.fromRuleContext(ruleContext).getJvmOptions())
          .addTransitiveInputs(JavaHelper.getHostJavabaseInputs(ruleContext));
    } else {
      builder.setExecutable(singleJar);
    }
    return builder;
  }
}
