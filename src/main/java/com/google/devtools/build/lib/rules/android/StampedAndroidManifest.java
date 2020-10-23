// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Type;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** An {@link AndroidManifest} stamped with the correct package. */
@Immutable
public class StampedAndroidManifest extends AndroidManifest {

  StampedAndroidManifest(Artifact manifest, @Nullable String pkg, boolean exported) {
    super(manifest, pkg, exported);
  }

  @Override
  public StampedAndroidManifest stamp(AndroidDataContext dataContext) {
    // This manifest is already stamped
    return this;
  }

  /**
   * Gets the manifest artifact wrapped by this object.
   *
   * <p>The manifest is guaranteed to be stamped with the correct Android package.
   */
  @Override
  public Artifact getManifest() {
    return super.getManifest();
  }

  ProcessedAndroidManifest withProcessedManifest(Artifact processedManifest) {
    return new ProcessedAndroidManifest(processedManifest, getPackage(), isExported());
  }

  /** Creates an empty manifest stamped with the default Java package for this target. */
  public static StampedAndroidManifest createEmpty(RuleContext ruleContext, boolean exported) {
    return createEmpty(ruleContext, AndroidCommon.getJavaPackage(ruleContext), exported);
  }

  /** Creates an empty manifest stamped with a specified package. */
  public static StampedAndroidManifest createEmpty(
      ActionConstructionContext context, String pkg, boolean exported) {
    Artifact generatedManifest =
        context.getUniqueDirectoryArtifact("_generated", "AndroidManifest.xml");

    String contents =
        Joiner.on("\n")
            .join(
                "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
                "<manifest xmlns:android=\"http://schemas.android.com/apk/res/android\"",
                "          package=\"" + pkg + "\">",
                "   <application>",
                "   </application>",
                "</manifest>");
    context.registerAction(
        FileWriteAction.create(context, generatedManifest, contents, /*makeExecutable=*/ false));
    return new StampedAndroidManifest(generatedManifest, pkg, exported);
  }

  public StampedAndroidManifest addMobileInstallStubApplication(RuleContext ruleContext)
      throws InterruptedException {

    Artifact stubManifest =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_MANIFEST);
    Artifact stubData =
        ruleContext.getImplicitOutputArtifact(
            AndroidRuleClasses.MOBILE_INSTALL_STUB_APPLICATION_DATA);

    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(ruleContext.getExecutablePrerequisite("$stubify_manifest"))
            .setProgressMessage("Injecting mobile install stub application")
            .setMnemonic("InjectMobileInstallStubApplication")
            .addInput(getManifest())
            .addOutput(stubManifest)
            .addOutput(stubData);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .add("--mode=mobile_install")
            .addExecPath("--input_manifest", getManifest())
            .addExecPath("--output_manifest", stubManifest)
            .addExecPath("--output_datafile", stubData);

    String overridePackage = getManifestValues(ruleContext).get("applicationId");
    if (overridePackage != null) {
      commandLine.add("--override_package", overridePackage);
    }

    builder.addCommandLine(commandLine.build());
    ruleContext.registerAction(builder.build(ruleContext));

    return new StampedAndroidManifest(stubManifest, getPackage(), isExported());
  }

  public static Map<String, String> getManifestValues(RuleContext context) {
    if (!context.attributes().isAttributeValueExplicitlySpecified("manifest_values")) {
      return ImmutableMap.of();
    }

    Map<String, String> manifestValues =
        new TreeMap<>(context.attributes().get("manifest_values", Type.STRING_DICT));

    for (String variable : manifestValues.keySet()) {
      manifestValues.put(
          variable, context.getExpander().expand("manifest_values", manifestValues.get(variable)));
    }
    return ImmutableMap.copyOf(manifestValues);
  }

  public StampedAndroidManifest createSplitManifest(
      RuleContext ruleContext, String splitName, boolean hasCode) {
    // aapt insists that manifests be called AndroidManifest.xml, even though they have to be
    // explicitly designated as manifests on the command line
    Artifact splitManifest =
        AndroidBinary.getDxArtifact(ruleContext, "split_" + splitName + "/AndroidManifest.xml");
    SpawnAction.Builder builder =
        new SpawnAction.Builder()
            .setExecutable(ruleContext.getExecutablePrerequisite("$build_split_manifest"))
            .setProgressMessage("Creating manifest for split %s", splitName)
            .setMnemonic("AndroidBuildSplitManifest")
            .addInput(getManifest())
            .addOutput(splitManifest);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addExecPath("--main_manifest", getManifest())
            .addExecPath("--split_manifest", splitManifest)
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

    return new StampedAndroidManifest(splitManifest, getPackage(), isExported());
  }

  public AndroidManifestInfo toProvider() {
    return AndroidManifestInfo.of(getManifest(), getPackage(), isExported());
  }

  @Override
  public boolean equals(Object object) {
    return (object instanceof StampedAndroidManifest && super.equals(object));
  }
}
