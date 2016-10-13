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
package com.google.devtools.build.lib.bazel.rules.android;

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
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.android.ApplicationManifest;
import com.google.devtools.build.lib.rules.android.LocalResourceContainer;
import com.google.devtools.build.lib.rules.android.ResourceApk;
import com.google.devtools.build.lib.rules.android.ResourceDependencies;
import com.google.devtools.build.lib.rules.java.JavaCommon;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider;
import com.google.devtools.build.lib.rules.java.JavaRuleOutputJarsProvider.OutputJar;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * An implementation for the aar_import rule.
 *
 * AAR files are zip archives that contain an Android Manifest, JARs, resources, assets, native
 * libraries, Proguard configuration and lint jars. Currently the aar_import rule supports AARs with
 * an AndroidManifest.xml, classes.jar and res/. Assets, native libraries and additional embedded
 * jars are not yet supported.
 *
 * @see <a href="http://tools.android.com/tech-docs/new-build-system/aar-format">AAR Format</a>
 */
public class AarImport implements RuleConfiguredTargetFactory {
  private static final String ANDROID_MANIFEST = "AndroidManifest.xml";
  private static final String CLASSES_JAR = "classes.jar";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    RuleConfiguredTargetBuilder ruleBuilder = new RuleConfiguredTargetBuilder(ruleContext);
    Artifact aar = ruleContext.getPrerequisiteArtifact("aar", Mode.TARGET);

    // classes.jar is required in every AAR.
    Artifact classesJar = createAarArtifact(ruleContext, CLASSES_JAR);
    ruleContext.registerAction(
        createEmbeddedJarExtractorActions(ruleContext, aar, CLASSES_JAR, classesJar));

    // AndroidManifest.xml is required in every AAR.
    Artifact androidManifestArtifact = createAarArtifact(ruleContext, ANDROID_MANIFEST);
    ruleContext.registerAction(createSingleFileExtractorActions(
        ruleContext, aar, ANDROID_MANIFEST, androidManifestArtifact));

    Artifact resourcesManifest = createAarArtifact(ruleContext, "resource_manifest");
    ruleContext.registerAction(
        createManifestExtractorActions(ruleContext, aar, "res/.*", resourcesManifest));

    Artifact resources = createResourcesTreeArtifact(ruleContext);
    ruleContext.registerAction(
        createManifestFileEntriesExtractorActions(ruleContext, aar, resourcesManifest, resources));

    ApplicationManifest androidManifest =
        ApplicationManifest.fromExplicitManifest(ruleContext, androidManifestArtifact);

    FileProvider resourcesProvider = new FileProvider(
        new NestedSetBuilder<Artifact>(Order.NAIVE_LINK_ORDER).add(resources).build());

    ResourceApk resourceApk = androidManifest.packWithDataAndResources(
        ruleContext,
        new LocalResourceContainer.Builder(ruleContext)
            .withResources(ImmutableList.of(resourcesProvider))
            .build(),
        ResourceDependencies.fromRuleDeps(ruleContext, JavaCommon.isNeverLink(ruleContext)),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_R_TXT),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_SYMBOLS_TXT),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_PROCESSED_MANIFEST),
        ruleContext.getImplicitOutputArtifact(AndroidRuleClasses.ANDROID_RESOURCES_ZIP));

    NestedSetBuilder<Artifact> filesToBuildBuilder =
        NestedSetBuilder.<Artifact>stableOrder().add(resources).add(classesJar);

    JavaRuleOutputJarsProvider.Builder jarProviderBuilder = new JavaRuleOutputJarsProvider.Builder()
        .addOutputJar(classesJar, null, null);
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

  // Extracts a jar file from the aar if it exists, otherwise outputs an empty jar file.
  private static Action[] createEmbeddedJarExtractorActions(RuleContext ruleContext, Artifact aar,
      String filename, Artifact outputArtifact) {
    return new SpawnAction.Builder()
        .setExecutable(ruleContext.getExecutablePrerequisite("$embedded_jar_extractor", Mode.HOST))
        .setMnemonic("AarJarExtractor")
        .setProgressMessage("Extracting " + filename + " from " + aar.getFilename())
        .addArgument("--input_archive")
        .addInputArgument(aar)
        .addArgument("--filename")
        .addArgument(filename)
        .addArgument("--output_dir")
        .addOutput(outputArtifact)
        .addArgument(outputArtifact.getExecPath().getParentDirectory().getPathString())
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

  private static Artifact createAarArtifact(RuleContext ruleContext, String name) {
    return ruleContext.getUniqueDirectoryArtifact(
        "_aar", name, ruleContext.getBinOrGenfilesDirectory());
  }

  private static Artifact createResourcesTreeArtifact(RuleContext ruleContext) {
    PathFragment rootRelativePath = ruleContext.getUniqueDirectory("_aar/unzipped");
    return ruleContext.getTreeArtifact(rootRelativePath, ruleContext.getBinOrGenfilesDirectory());
  }
}
