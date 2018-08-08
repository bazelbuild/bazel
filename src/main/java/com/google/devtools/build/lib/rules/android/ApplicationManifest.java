// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidManifestMerger;
import com.google.devtools.build.lib.syntax.Type;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** Represents a AndroidManifest, that may have been merged from dependencies. */
public final class ApplicationManifest {

  static Artifact createSplitManifest(
      RuleContext ruleContext, Artifact manifest, String splitName, boolean hasCode) {
    // aapt insists that manifests be called AndroidManifest.xml, even though they have to be
    // explicitly designated as manifests on the command line
    Artifact result =
        AndroidBinary.getDxArtifact(ruleContext, "split_" + splitName + "/AndroidManifest.xml");
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$build_split_manifest", Mode.HOST))
            .setProgressMessage("Creating manifest for split %s", splitName)
            .setMnemonic("AndroidBuildSplitManifest")
            .addInput(manifest)
            .addOutput(result);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addExecPath("--main_manifest", manifest)
            .addExecPath("--split_manifest", result)
            .add("--split", splitName);
    if (hasCode) {
      commandLine.add("--hascode");
    } else {
      commandLine.add("--nohascode");
    }

    String overridePackage = getManifestValues(ruleContext).get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));
    return result;
  }

  static Artifact addMobileInstallStubApplication(RuleContext ruleContext, Artifact manifest)
      throws InterruptedException {

    Artifact stubManifest =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_MANIFEST);
    Artifact stubData =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_DATA);

    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest", Mode.HOST))
            .setProgressMessage("Injecting mobile install stub application")
            .setMnemonic("InjectMobileInstallStubApplication")
            .addInput(manifest)
            .addOutput(stubManifest)
            .addOutput(stubData);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .add("--mode=mobile_install")
            .addExecPath("--input_manifest", manifest)
            .addExecPath("--output_manifest", stubManifest)
            .addExecPath("--output_datafile", stubData);

    String overridePackage = getManifestValues(ruleContext).get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));

    return stubManifest;
  }

  public static Artifact getManifestFromAttributes(RuleContext ruleContext) {
    return ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET);
  }

  static Artifact renameManifestIfNeeded(AndroidDataContext dataContext, Artifact manifest)
      throws InterruptedException {
    if (manifest.getFilename().equals("AndroidManifest.xml")) {
      return manifest;
    } else {
      /*
       * If the manifest file is not named AndroidManifest.xml, we create a symlink named
       * AndroidManifest.xml to it. aapt requires the manifest to be named as such.
       */
      Artifact manifestSymlink =
          dataContext.createOutputArtifact(AndroidRuleClasses.ANDROID_SYMLINKED_MANIFEST);
      dataContext.registerAction(
          new SymlinkAction(
              dataContext.getActionConstructionContext().getActionOwner(),
              manifest,
              manifestSymlink,
              "Renaming Android manifest for " + dataContext.getLabel()));
      return manifestSymlink;
    }
  }

  /**
   * Creates an action to generate an empty manifest file with a specific package name.
   *
   * @return an artifact for the generated manifest
   */
  public static Artifact generateManifest(
      ActionConstructionContext context, String manifestPackage) {
    Artifact generatedManifest =
        context.getUniqueDirectoryArtifact("_generated", "AndroidManifest.xml");

    String contents =
        Joiner.on("\n")
            .join(
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\"",
                "          package=\"" + manifestPackage + "\">",
                "   <application>",
                "   </application>",
                "</manifest>");
    context.registerAction(
        FileWriteAction.create(context, generatedManifest, contents, /*makeExecutable=*/ false));
    return generatedManifest;
  }

  /** Gets a map of manifest values from this rule's 'manifest_values' attribute */
  public static ImmutableMap<String, String> getManifestValues(RuleContext context) {
    Map<String, String> manifestValues = new TreeMap<>();
    if (context.attributes().isAttributeValueExplicitlySpecified("manifest_values")) {
      manifestValues.putAll(context.attributes().get("manifest_values", Type.STRING_DICT));
    }

    for (String variable : manifestValues.keySet()) {
      manifestValues.put(
          variable, context.getExpander().expand("manifest_values", manifestValues.get(variable)));
    }
    return ImmutableMap.copyOf(manifestValues);
  }

  private ApplicationManifest() {}

  static Optional<Artifact> maybeMergeWith(
      AndroidDataContext dataContext,
      AndroidSemantics androidSemantics,
      Artifact primaryManifest,
      ResourceDependencies resourceDeps,
      Map<String, String> manifestValues,
      boolean useLegacyMerging,
      String customPackage) {
    Map<Artifact, Label> mergeeManifests = getMergeeManifests(resourceDeps.getResourceContainers());

    if (useLegacyMerging) {
      return androidSemantics.maybeDoLegacyManifestMerging(
          mergeeManifests, dataContext, primaryManifest);
    } else {
      if (!mergeeManifests.isEmpty() || !manifestValues.isEmpty()) {
        Artifact outputManifest =
            dataContext.getUniqueDirectoryArtifact("_merged", "AndroidManifest.xml");
        Artifact mergeLog =
            dataContext.getUniqueDirectoryArtifact("_merged", "manifest_merger_log.txt");
        new ManifestMergerActionBuilder()
            .setManifest(primaryManifest)
            .setMergeeManifests(mergeeManifests)
            .setLibrary(false)
            .setManifestValues(manifestValues)
            .setCustomPackage(customPackage)
            .setManifestOutput(outputManifest)
            .setLogOut(mergeLog)
            .build(dataContext);
        return Optional.of(outputManifest);
      }
    }
    return Optional.empty();
  }

  /** Checks if the legacy manifest merger should be used, based on a rule attribute */
  public static boolean useLegacyMerging(RuleContext ruleContext) {
    return ruleContext.isLegalFragment(AndroidConfiguration.class)
        && ruleContext.getRule().isAttrDefined("manifest_merger", STRING)
        && useLegacyMerging(
            ruleContext,
            AndroidCommon.getAndroidConfig(ruleContext),
            ruleContext.attributes().get("manifest_merger", STRING));
  }

  /**
   * Checks if the legacy manifest merger should be used, based on an optional string specifying the
   * merger to use.
   */
  public static boolean useLegacyMerging(
      RuleErrorConsumer errorConsumer,
      AndroidConfiguration androidConfig,
      @Nullable String mergerString) {
    AndroidManifestMerger merger = AndroidManifestMerger.fromString(mergerString);
    if (merger == null) {
      merger = androidConfig.getManifestMerger();
    }
    if (merger == AndroidManifestMerger.LEGACY) {
      errorConsumer.ruleWarning(
          "manifest_merger 'legacy' is deprecated. Please update to 'android'.\n"
              + "See https://developer.android.com/studio/build/manifest-merge.html for more "
              + "information about the manifest merger.");
    }

    return merger == AndroidManifestMerger.LEGACY;
  }

  private static Map<Artifact, Label> getMergeeManifests(
      Iterable<ValidatedAndroidData> transitiveData) {
    ImmutableSortedMap.Builder<Artifact, Label> builder =
        ImmutableSortedMap.orderedBy(Artifact.EXEC_PATH_COMPARATOR);
    for (ValidatedAndroidData d : transitiveData) {
      if (d.isManifestExported()) {
        builder.put(d.getManifest(), d.getLabel());
      }
    }
    return builder.build();
  }

  static Optional<Artifact> maybeSetManifestPackage(
      AndroidDataContext dataContext, Artifact manifest, String customPackage) {
    if (isNullOrEmpty(customPackage)) {
      return Optional.empty();
    }
    Artifact outputManifest =
        dataContext.getUniqueDirectoryArtifact("_renamed", "AndroidManifest.xml");
    new ManifestMergerActionBuilder()
        .setManifest(manifest)
        .setLibrary(true)
        .setCustomPackage(customPackage)
        .setManifestOutput(outputManifest)
        .build(dataContext);

    return Optional.of(outputManifest);
  }
}
