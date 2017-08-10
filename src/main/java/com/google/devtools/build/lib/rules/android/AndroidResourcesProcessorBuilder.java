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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.Builder;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.ResourceContainerConverter.Builder.SeparatorType;
import com.google.devtools.build.lib.util.OS;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Builder for creating resource processing action.
 */
public class AndroidResourcesProcessorBuilder {

  private static final ResourceContainerConverter.ToArtifacts AAPT2_RESOURCE_DEP_TO_ARTIFACTS =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .includeAapt2RTxt()
          .includeSymbolsBin()
          .includeStaticLibrary()
          .toArtifactConverter();

  private static final ResourceContainerConverter.ToArg AAPT2_RESOURCE_DEP_TO_ARG =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .includeAapt2RTxt()
          .includeSymbolsBin()
          .includeStaticLibrary()
          .withSeparator(SeparatorType.COLON_COMMA)
          .toArgConverter();

  private static final ResourceContainerConverter.ToArtifacts RESOURCE_CONTAINER_TO_ARTIFACTS =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .toArtifactConverter();

  private static final ResourceContainerConverter.ToArtifacts RESOURCE_DEP_TO_ARTIFACTS =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .includeRTxt()
          .includeSymbolsBin()
          .toArtifactConverter();

  private static final ResourceContainerConverter.ToArg RESOURCE_CONTAINER_TO_ARG =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .withSeparator(SeparatorType.COLON_COMMA)
          .toArgConverter();

  private static final ResourceContainerConverter.ToArg RESOURCE_DEP_TO_ARG =
      ResourceContainerConverter.builder()
          .includeResourceRoots()
          .includeManifest()
          .includeRTxt()
          .includeSymbolsBin()
          .withSeparator(SeparatorType.COLON_COMMA)
          .toArgConverter();

  private ResourceContainer primary;
  private ResourceDependencies dependencies;
  private Artifact proguardOut;
  private Artifact mainDexProguardOut;
  private Artifact rTxtOut;
  private Artifact sourceJarOut;
  private boolean debug = false;
  private ResourceFilter resourceFilter;
  private List<String> uncompressedExtensions = Collections.emptyList();
  private Artifact apkOut;
  private final AndroidSdkProvider sdk;
  private List<String> assetsToIgnore = Collections.emptyList();
  private SpawnAction.Builder spawnActionBuilder;
  private String customJavaPackage;
  private final RuleContext ruleContext;
  private String versionCode;
  private String applicationId;
  private String versionName;
  private Artifact symbols;
  private Artifact dataBindingInfoZip;

  private Artifact manifestOut;
  private Artifact mergedResourcesOut;
  private boolean isLibrary;
  private boolean crunchPng = true;
  private Artifact featureOf;
  private Artifact featureAfter;
  private AndroidAaptVersion aaptVersion;
  private boolean throwOnResourceConflict;

  /**
   * @param ruleContext The RuleContext that was used to create the SpawnAction.Builder.
   */
  public AndroidResourcesProcessorBuilder(RuleContext ruleContext) {
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    this.ruleContext = ruleContext;
    this.spawnActionBuilder = new SpawnAction.Builder();
    this.resourceFilter = ResourceFilter.empty(ruleContext);
  }

  /**
   * The primary resource for merging. This resource will overwrite any resource or data
   * value in the transitive closure.
   */
  public AndroidResourcesProcessorBuilder withPrimary(ResourceContainer primary) {
    this.primary = primary;
    return this;
  }

  /**
   * The output zip for resource-processed data binding expressions (i.e. a zip of .xml files).
   * If null, data binding processing is skipped (and data binding expressions aren't allowed in
   * layout resources).
   */
  public AndroidResourcesProcessorBuilder setDataBindingInfoZip(Artifact zip) {
    this.dataBindingInfoZip = zip;
    return this;
  }

  public AndroidResourcesProcessorBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencies = resourceDeps;
    return this;
  }

  public AndroidResourcesProcessorBuilder setUncompressedExtensions(
      List<String> uncompressedExtensions) {
    this.uncompressedExtensions = uncompressedExtensions;
    return this;
  }

  public AndroidResourcesProcessorBuilder setCrunchPng(boolean crunchPng) {
    this.crunchPng = crunchPng;
    return this;
  }

  public AndroidResourcesProcessorBuilder setResourceFilter(
      ResourceFilter resourceFilter) {
    this.resourceFilter = resourceFilter;
    return this;
  }

  public AndroidResourcesProcessorBuilder setDebug(boolean debug) {
    this.debug = debug;
    return this;
  }

  public AndroidResourcesProcessorBuilder setProguardOut(Artifact proguardCfg) {
    this.proguardOut = proguardCfg;
    return this;
  }

  public AndroidResourcesProcessorBuilder setMainDexProguardOut(Artifact mainDexProguardCfg) {
    this.mainDexProguardOut = mainDexProguardCfg;
    return this;
  }

  public AndroidResourcesProcessorBuilder setRTxtOut(Artifact rTxtOut) {
    this.rTxtOut = rTxtOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setSymbols(Artifact symbols) {
    this.symbols = symbols;
    return this;
  }

  public AndroidResourcesProcessorBuilder setApkOut(Artifact apkOut) {
    this.apkOut = apkOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setSourceJarOut(Artifact sourceJarOut) {
    this.sourceJarOut = sourceJarOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setAssetsToIgnore(List<String> assetsToIgnore) {
    this.assetsToIgnore = assetsToIgnore;
    return this;
  }

  public AndroidResourcesProcessorBuilder setManifestOut(Artifact manifestOut) {
    this.manifestOut = manifestOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setMergedResourcesOut(Artifact mergedResourcesOut) {
    this.mergedResourcesOut = mergedResourcesOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setLibrary(boolean isLibrary) {
    this.isLibrary = isLibrary;
    return this;
  }

  public AndroidResourcesProcessorBuilder setFeatureOf(Artifact featureOf) {
    this.featureOf = featureOf;
    return this;
  }

  public AndroidResourcesProcessorBuilder setFeatureAfter(Artifact featureAfter) {
    this.featureAfter = featureAfter;
    return this;
  }

  public AndroidResourcesProcessorBuilder targetAaptVersion(AndroidAaptVersion aaptVersion) {
    this.aaptVersion = aaptVersion;
    return this;
  }

  public AndroidResourcesProcessorBuilder setThrowOnResourceConflict(
      boolean throwOnResourceConflict) {
    this.throwOnResourceConflict = throwOnResourceConflict;
    return this;
  }

  public ResourceContainer build(ActionConstructionContext context) {
    if (aaptVersion == AndroidAaptVersion.AAPT2) {
      return createAapt2ApkAction(context);
    }
    return createAaptAction(context);
  }

  public AndroidResourcesProcessorBuilder setJavaPackage(String customJavaPackage) {
    this.customJavaPackage = customJavaPackage;
    return this;
  }

  public AndroidResourcesProcessorBuilder setVersionCode(String versionCode) {
    this.versionCode = versionCode;
    return this;
  }

  public AndroidResourcesProcessorBuilder setApplicationId(String applicationId) {
    if (applicationId != null && !applicationId.isEmpty()) {
      this.applicationId = applicationId;
    }
    return this;
  }

  public AndroidResourcesProcessorBuilder setVersionName(String versionName) {
    this.versionName = versionName;
    return this;
  }

  private ResourceContainer createAapt2ApkAction(ActionConstructionContext context) {
    List<Artifact> outs = new ArrayList<>();
    // TODO(corysmith): Convert to an immutable list builder, as there is no benefit to a NestedSet
    // here, as it will already have been flattened.
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("AAPT2_PACKAGE").add("--");

    builder.add("--aapt2", sdk.getAapt2().getExecutable());
    ResourceContainerConverter.convertDependencies(
        dependencies, builder, inputs, AAPT2_RESOURCE_DEP_TO_ARG, AAPT2_RESOURCE_DEP_TO_ARTIFACTS);

    configureCommonFlags(outs, inputs, builder);

    if (OS.getCurrent() == OS.WINDOWS) {
      // Some flags (e.g. --mainData) may specify lists (or lists of lists) separated by special
      // characters (colon, semicolon, hashmark, ampersand) that don't work on Windows, and quoting
      // semantics are very complicated (more so than in Bash), so let's just always use a parameter
      // file.
      // TODO(laszlocsomor), TODO(corysmith): restructure the Android BusyBux's flags by deprecating
      // list-type and list-of-list-type flags that use such problematic separators in favor of
      // multi-value flags (to remove one level of listing) and by changing all list separators to a
      // platform-safe character (= comma).
      this.spawnActionBuilder.alwaysUseParameterFile(ParameterFileType.UNQUOTED);
    } else {
      this.spawnActionBuilder.useParameterFile(ParameterFileType.UNQUOTED);
    }

    // Create the spawn action.
    ruleContext.registerAction(
        this.spawnActionBuilder
            .useDefaultShellEnvironment()
            .useDefaultShellEnvironment()
            .addTool(sdk.getAapt2())
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Processing Android resources for %s", ruleContext.getLabel())
            .setMnemonic("AndroidAapt2")
            .build(context));

    // Return the full set of processed transitive dependencies.
    ResourceContainer.Builder result =
        primary.toBuilder().setJavaSourceJar(sourceJarOut).setRTxt(rTxtOut).setSymbols(symbols);
    // If there is an apk to be generated, use it, else reuse the apk from the primary resources.
    // All android_binary ResourceContainers have to have an apk, but if a new one is not
    // requested to be built for this resource processing action (in case of just creating an
    // R.txt or proguard merging), reuse the primary resource from the dependencies.
    if (apkOut != null) {
      result.setApk(apkOut);
    }
    if (manifestOut != null) {
      result.setManifest(manifestOut);
    }
    return result.build();
  }

  private ResourceContainer createAaptAction(ActionConstructionContext context) {
    List<Artifact> outs = new ArrayList<>();
    // TODO(corysmith): Convert to an immutable list builder, as there is no benefit to a NestedSet
    // here, as it will already have been flattened.
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    // Set the busybox tool.
    builder.add("--tool").add("PACKAGE").add("--");

    ResourceContainerConverter.convertDependencies(
        dependencies, builder, inputs, RESOURCE_DEP_TO_ARG, RESOURCE_DEP_TO_ARTIFACTS);
    builder.add("--aapt", sdk.getAapt().getExecutable());
    configureCommonFlags(outs, inputs, builder);

    if (OS.getCurrent() == OS.WINDOWS) {
      // Some flags (e.g. --mainData) may specify lists (or lists of lists) separated by special
      // characters (colon, semicolon, hashmark, ampersand) that don't work on Windows, and quoting
      // semantics are very complicated (more so than in Bash), so let's just always use a parameter
      // file.
      // TODO(laszlocsomor), TODO(corysmith): restructure the Android BusyBux's flags by deprecating
      // list-type and list-of-list-type flags that use such problematic separators in favor of
      // multi-value flags (to remove one level of listing) and by changing all list separators to a
      // platform-safe character (= comma).
      this.spawnActionBuilder.alwaysUseParameterFile(ParameterFileType.UNQUOTED);
    } else {
      this.spawnActionBuilder.useParameterFile(ParameterFileType.UNQUOTED);
    }

    // Create the spawn action.
    ruleContext.registerAction(
        this.spawnActionBuilder
            .useDefaultShellEnvironment()
            .useDefaultShellEnvironment()
            .addTool(sdk.getAapt())
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Processing Android resources for %s", ruleContext.getLabel())
            .setMnemonic("AaptPackage")
            .build(context));

    // Return the full set of processed transitive dependencies.
    ResourceContainer.Builder result =
        primary.toBuilder().setJavaSourceJar(sourceJarOut).setRTxt(rTxtOut).setSymbols(symbols);
    // If there is an apk to be generated, use it, else reuse the apk from the primary resources.
    // All android_binary ResourceContainers have to have an apk, but if a new one is not
    // requested to be built for this resource processing action (in case of just creating an
    // R.txt or proguard merging), reuse the primary resource from the dependencies.
    if (apkOut != null) {
      result.setApk(apkOut);
    }
    if (manifestOut != null) {
      result.setManifest(manifestOut);
    }
    return result.build();
  }

  private void configureCommonFlags(
      List<Artifact> outs, NestedSetBuilder<Artifact> inputs, Builder builder) {

    // Add data
    builder.add("--primaryData").add(RESOURCE_CONTAINER_TO_ARG.apply(primary));
    inputs.addTransitive(RESOURCE_CONTAINER_TO_ARTIFACTS.apply(primary));

    if (!Strings.isNullOrEmpty(sdk.getBuildToolsVersion())) {
      builder.add("--buildToolsVersion").add(sdk.getBuildToolsVersion());
    }

    builder.add("--annotationJar", sdk.getAnnotationsJar());
    inputs.add(sdk.getAnnotationsJar());

    builder.add("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    if (isLibrary) {
      builder.add("--packageType").add("LIBRARY");
    }

    if (rTxtOut != null) {
      builder.add("--rOutput", rTxtOut);
      outs.add(rTxtOut);
    }

    if (symbols != null) {
      builder.add("--symbolsOut", symbols);
      outs.add(symbols);
    }
    if (sourceJarOut != null) {
      builder.add("--srcJarOutput", sourceJarOut);
      outs.add(sourceJarOut);
    }
    if (proguardOut != null) {
      builder.add("--proguardOutput", proguardOut);
      outs.add(proguardOut);
    }

    if (mainDexProguardOut != null) {
      builder.add("--mainDexProguardOutput", mainDexProguardOut);
      outs.add(mainDexProguardOut);
    }

    if (manifestOut != null) {
      builder.add("--manifestOutput", manifestOut);
      outs.add(manifestOut);
    }

    if (mergedResourcesOut != null) {
      builder.add("--resourcesOutput", mergedResourcesOut);
      outs.add(mergedResourcesOut);
    }

    if (apkOut != null) {
      builder.add("--packagePath", apkOut);
      outs.add(apkOut);
    }
    if (resourceFilter.hasConfigurationFilters() && !resourceFilter.isPrefiltering()) {
      builder.add("--resourceConfigs").add(resourceFilter.getConfigurationFilterString());
    }
    if (resourceFilter.hasDensities() && !resourceFilter.isPrefiltering()) {
      builder.add("--densities").add(resourceFilter.getDensityString());
    }
    ImmutableList<String> filteredResources = resourceFilter.getResourcesToIgnoreInExecution();
    if (!filteredResources.isEmpty()) {
      builder.add("--prefilteredResources", VectorArg.of(filteredResources).joinWith(","));
    }
    if (!uncompressedExtensions.isEmpty()) {
      builder.add(
          "--uncompressedExtensions",
          VectorArg.of(ImmutableList.copyOf(uncompressedExtensions)).joinWith(","));
    }
    if (!crunchPng) {
      builder.add("--useAaptCruncher=no");
    }
    if (!assetsToIgnore.isEmpty()) {
      builder.add(
          "--assetsToIgnore", VectorArg.of(ImmutableList.copyOf(assetsToIgnore)).joinWith(","));
    }
    if (debug) {
      builder.add("--debug");
    }

    if (versionCode != null) {
      builder.add("--versionCode").add(versionCode);
    }

    if (versionName != null) {
      builder.add("--versionName").add(versionName);
    }

    if (applicationId != null) {
      builder.add("--applicationId").add(applicationId);
    }

    if (dataBindingInfoZip != null) {
      builder.add("--dataBindingInfoOut", dataBindingInfoZip);
      outs.add(dataBindingInfoZip);
    }

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR").add(customJavaPackage);
    }

    if (featureOf != null) {
      builder.add("--featureOf", featureOf);
      inputs.add(featureOf);
    }

    if (featureAfter != null) {
      builder.add("--featureAfter", featureAfter);
      inputs.add(featureAfter);
    }

    if (throwOnResourceConflict) {
      builder.add("--throwOnResourceConflict");
    }
  }
}
