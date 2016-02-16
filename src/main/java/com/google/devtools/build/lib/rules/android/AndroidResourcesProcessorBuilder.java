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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Builder for creating resource processing action.
 */
public class AndroidResourcesProcessorBuilder {
  private static final ResourceContainerToArtifacts RESOURCE_CONTAINER_TO_ARTIFACTS =
      new ResourceContainerToArtifacts(false);

  private static final ResourceContainerToArtifacts RESOURCE_DEP_TO_ARTIFACTS =
      new ResourceContainerToArtifacts(true);

  private static final ResourceContainerToArg RESOURCE_CONTAINER_TO_ARG =
      new ResourceContainerToArg(false);

  private static final ResourceContainerToArg RESOURCE_DEP_TO_ARG =
      new ResourceContainerToArg(true);
  private ResourceContainer primary;
  private ResourceDependencies dependencies;
  private Artifact proguardOut;
  private Artifact rTxtOut;
  private Artifact sourceJarOut;
  private boolean debug = false;
  private List<String> resourceConfigs = Collections.emptyList();
  private List<String> uncompressedExtensions = Collections.emptyList();
  private Artifact apkOut;
  private final AndroidSdkProvider sdk;
  private List<String> assetsToIgnore = Collections.emptyList();
  private SpawnAction.Builder spawnActionBuilder;
  private List<String> densities = Collections.emptyList();
  private String customJavaPackage;
  private final RuleContext ruleContext;
  private String versionCode;
  private String applicationId;
  private String versionName;
  private Artifact symbolsTxt;

  private Artifact manifestOut;

  /**
   * @param ruleContext The RuleContext that was used to create the SpawnAction.Builder.
   */
  public AndroidResourcesProcessorBuilder(RuleContext ruleContext) {
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    this.ruleContext = ruleContext;
    this.spawnActionBuilder = new SpawnAction.Builder();
  }

  /**
   * The primary resource for merging. This resource will overwrite any resource or data
   * value in the transitive closure.
   */
  public AndroidResourcesProcessorBuilder withPrimary(ResourceContainer primary) {
    this.primary = primary;
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

  public AndroidResourcesProcessorBuilder setDensities(List<String> densities) {
    this.densities = densities;
    return this;
  }

  public AndroidResourcesProcessorBuilder setConfigurationFilters(List<String> resourceConfigs) {
    this.resourceConfigs = resourceConfigs;
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

  public AndroidResourcesProcessorBuilder setRTxtOut(Artifact rTxtOut) {
    this.rTxtOut = rTxtOut;
    return this;
  }

  public AndroidResourcesProcessorBuilder setSymbolsTxt(Artifact symbolsTxt) {
    this.symbolsTxt = symbolsTxt;
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

  private static class ResourceContainerToArg implements Function<ResourceContainer, String> {
    private boolean includeSymbols;

    public ResourceContainerToArg(boolean includeSymbols) {
      this.includeSymbols = includeSymbols;
    }

    @Override
    public String apply(ResourceContainer container) {
      StringBuilder builder = new StringBuilder();
      builder.append(convertRoots(container, ResourceType.RESOURCES))
          .append(":")
          .append(convertRoots(container, ResourceType.ASSETS))
          .append(":")
          .append(container.getManifest().getExecPathString());
      if (includeSymbols) {
        builder.append(":")
            .append(container.getRTxt() == null ? "" : container.getRTxt().getExecPath())
            .append(":")
            .append(
                container.getSymbolsTxt() == null ? "" : container.getSymbolsTxt().getExecPath());
      }
      return builder.toString();
    }
  }

  private static class ResourceContainerToArtifacts
      implements Function<ResourceContainer, NestedSet<Artifact>> {

    private boolean includeSymbols;

    public ResourceContainerToArtifacts(boolean includeSymbols) {
      this.includeSymbols = includeSymbols;
    }

    @Override
    public NestedSet<Artifact> apply(ResourceContainer container) {
      NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.naiveLinkOrder();
      addIfNotNull(container.getManifest(), artifacts);
      if (includeSymbols) {
        addIfNotNull(container.getRTxt(), artifacts);
        addIfNotNull(container.getSymbolsTxt(), artifacts);
      }
      artifacts.addAll(container.getArtifacts());
      return artifacts.build();
    }

    private void addIfNotNull(@Nullable Artifact artifact, NestedSetBuilder<Artifact> artifacts) {
      if (artifact != null) {
        artifacts.add(artifact);
      }
    }
  }

  @VisibleForTesting
  public static String convertRoots(ResourceContainer container, ResourceType resourceType) {
    return Joiner.on("#").join(Iterators.transform(
        container.getRoots(resourceType).iterator(), Functions.toStringFunction()));
  }

  public ResourceContainer build(ActionConstructionContext context) {
    List<Artifact> outs = new ArrayList<>();
    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();

    if (!Strings.isNullOrEmpty(sdk.getBuildToolsVersion())) {
      builder.add("--buildToolsVersion").add(sdk.getBuildToolsVersion());
    }

    builder.addExecPath("--aapt", sdk.getAapt().getExecutable());
    // Use a FluentIterable to avoid flattening the NestedSets
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    inputs.addAll(ruleContext.getExecutablePrerequisite("$android_resources_processor", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifactsWithoutMiddlemen());

    builder.addExecPath("--annotationJar", sdk.getAnnotationsJar());
    inputs.add(sdk.getAnnotationsJar());

    builder.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    builder.add("--primaryData").add(RESOURCE_CONTAINER_TO_ARG.apply(primary));
    inputs.addTransitive(RESOURCE_CONTAINER_TO_ARTIFACTS.apply(primary));

    if (dependencies != null) {
      // TODO(bazel-team): Find an appropriately lazy method to deduplicate the dependencies between
      // the direct and transitive data.
      // Add transitive data inside an unmodifiableIterable to ensure it won't be expanded until
      // iteration.
      if (!dependencies.getTransitiveResources().isEmpty()) {
        builder.addJoinStrings("--data", ",",
            Iterables.unmodifiableIterable(
                Iterables.transform(dependencies.getTransitiveResources(), RESOURCE_DEP_TO_ARG)));
      }
      // Add direct data inside an unmodifiableIterable to ensure it won't be expanded until
      // iteration.
      if (!dependencies.getDirectResources().isEmpty()) {
        builder.addJoinStrings("--directData", ",",
            Iterables.unmodifiableIterable(
                Iterables.transform(dependencies.getDirectResources(), RESOURCE_DEP_TO_ARG)));
      }
      // This flattens the nested set. Since each ResourceContainer needs to be transformed into 
      // Artifacts, and the NestedSetBuilder.wrap doesn't support lazy Iterator evaluation
      // and SpawnActionBuilder.addInputs evaluates Iterables, it becomes necessary to make the
      // best effort and let it get flattened.
      inputs.addTransitive(
          NestedSetBuilder.wrap(
              Order.NAIVE_LINK_ORDER,
              FluentIterable.from(dependencies.getResources())
                  .transformAndConcat(RESOURCE_DEP_TO_ARTIFACTS)));
    }

    if (rTxtOut != null) {
      builder.addExecPath("--rOutput", rTxtOut);
      outs.add(rTxtOut);
      // If R.txt is not null, dependency R.javas will not be regenerated from the R.txt found in
      // the deps, which means the resource processor needs to be told it is creating a library so
      // that it will generate the R.txt.
      builder.add("--packageType").add("LIBRARY");
    }

    if (symbolsTxt != null) {
      builder.addExecPath("--symbolsTxtOut", symbolsTxt);
      outs.add(symbolsTxt);
    }
    if (sourceJarOut != null) {
      builder.addExecPath("--srcJarOutput", sourceJarOut);
      outs.add(sourceJarOut);
    }
    if (proguardOut != null) {
      builder.addExecPath("--proguardOutput", proguardOut);
      outs.add(proguardOut);
    }
    
    if (manifestOut != null) {
      builder.addExecPath("--manifestOutput", manifestOut);
      outs.add(manifestOut);
    }
    
    if (apkOut != null) {
      builder.addExecPath("--packagePath", apkOut);
      outs.add(apkOut);
    }
    if (!resourceConfigs.isEmpty()) {
      builder.addJoinStrings("--resourceConfigs", ",", resourceConfigs);
    }
    if (!densities.isEmpty()) {
      builder.addJoinStrings("--densities", ",", densities);
    }
    if (!uncompressedExtensions.isEmpty()) {
      builder.addJoinStrings("--uncompressedExtensions", ",", uncompressedExtensions);
    }
    if (!assetsToIgnore.isEmpty()) {
      builder.addJoinStrings("--assetsToIgnore", ",", assetsToIgnore);
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

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this is allows android rules to generate resources outside of the java{,tests} tree.
      builder.add("--packageForR").add(customJavaPackage);
    }

    // Create the spawn action.
    ruleContext.registerAction(
        this.spawnActionBuilder
            .addTool(sdk.getAapt())
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>copyOf(outs))
            .setCommandLine(builder.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_processor", Mode.HOST))
            .setProgressMessage("Processing resources")
            .setMnemonic("AndroidAapt")
            .build(context));

    // Return the full set of processed transitive dependencies.
    return new ResourceContainer(primary.getLabel(),
        primary.getJavaPackage(),
        primary.getRenameManifestPackage(),
        primary.getConstantsInlined(),
        // If there is no apk to be generated, use the apk from the primary resources.
        // All ResourceContainers have to have an apk, but if a new one is not requested to be built
        // for this resource processing action (in case of just creating an R.txt or
        // proguard merging), reuse the primary resource from the dependencies.
        apkOut != null ? apkOut : primary.getApk(),
        manifestOut != null ? manifestOut : primary.getManifest(),
        sourceJarOut,
        primary.getArtifacts(ResourceType.ASSETS),
        primary.getArtifacts(ResourceType.RESOURCES),
        primary.getRoots(ResourceType.ASSETS),
        primary.getRoots(ResourceType.RESOURCES),
        primary.isManifestExported(),
        rTxtOut,
        symbolsTxt);
  }

  public AndroidResourcesProcessorBuilder setJavaPackage(String newManifestPackage) {
    this.customJavaPackage = newManifestPackage;
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
}
