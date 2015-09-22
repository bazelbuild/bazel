// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceContainer;
import com.google.devtools.build.lib.rules.android.AndroidResourcesProvider.ResourceType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Builder for creating resource processing action.
 */
public class AndroidResourcesProcessorBuilder {

  private ResourceContainer primary;
  private List<ResourceContainer> dependencies = Collections.emptyList();
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

  public AndroidResourcesProcessorBuilder withDependencies(Iterable<ResourceContainer> nestedSet) {
    this.dependencies = ImmutableList.copyOf(nestedSet);
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

  private void addResourceContainer(List<Artifact> inputs, List<String> args,
      ResourceContainer container) {
    Iterables.addAll(inputs, container.getArtifacts());
    inputs.add(container.getManifest());
    inputs.add(container.getRTxt());

    args.add(String.format("%s:%s:%s:%s:%s",
        convertRoots(container, ResourceType.RESOURCES),
        convertRoots(container, ResourceType.ASSETS),
        container.getManifest().getExecPathString(),
        container.getRTxt() == null ? "" : container.getRTxt().getExecPath(),
        container.getSymbolsTxt() == null ? "" : container.getSymbolsTxt().getExecPath()
    ));
  }

  private void addPrimaryResourceContainer(List<Artifact> inputs, List<String> args,
      ResourceContainer container) {
    Iterables.addAll(inputs, container.getArtifacts());
    inputs.add(container.getManifest());

    // no R.txt, because it will be generated from this action.
    args.add(String.format("%s:%s:%s",
        convertRoots(container, ResourceType.RESOURCES),
        convertRoots(container, ResourceType.ASSETS),
        container.getManifest().getExecPathString()
    ));
  }

  @VisibleForTesting
  public static String convertRoots(ResourceContainer container, ResourceType resourceType) {
    return Joiner.on("#").join(
        Iterators.transform(
            container.getRoots(resourceType).iterator(), Functions.toStringFunction()));
  }

  public ResourceContainer build(ActionConstructionContext context) {
    List<Artifact> outs = new ArrayList<>();
    List<Artifact> ins = new ArrayList<>();
    List<String> args = new ArrayList<>();

    args.add("--aapt");
    args.add(sdk.getAapt().getExecutable().getExecPathString());

    Iterables.addAll(ins,
        ruleContext.getExecutablePrerequisite("$android_resources_processor", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifactsWithoutMiddlemen());

    args.add("--annotationJar");
    args.add(sdk.getAnnotationsJar().getExecPathString());
    ins.add(sdk.getAnnotationsJar());
    args.add("--androidJar");
    args.add(sdk.getAndroidJar().getExecPathString());
    ins.add(sdk.getAndroidJar());

    args.add("--primaryData");
    addPrimaryResourceContainer(ins, args, primary);
    if (!dependencies.isEmpty()) {
      args.add("--data");
      List<String> data = new ArrayList<>();
      for (ResourceContainer container : dependencies) {
        addResourceContainer(ins, data, container);
      }
      args.add(Joiner.on(",").join(data));
    }

    if (rTxtOut != null) {
      args.add("--rOutput");
      args.add(rTxtOut.getExecPathString());
      outs.add(rTxtOut);
      // If R.txt is not null, dependency R.javas will not be regenerated from the R.txt found in
      // the deps, which means the resource processor needs to be told it is creating a library so
      // that it will generate the R.txt.
      args.add("--packageType");
      args.add("LIBRARY");
    }
    if (symbolsTxt != null) {
      args.add("--symbolsTxtOut");
      args.add(symbolsTxt.getExecPathString());
      outs.add(symbolsTxt);
    }
    if (sourceJarOut != null) {
      args.add("--srcJarOutput");
      args.add(sourceJarOut.getExecPathString());
      outs.add(sourceJarOut);
    }
    if (proguardOut != null) {
      args.add("--proguardOutput");
      args.add(proguardOut.getExecPathString());
      outs.add(proguardOut);
    }
    if (apkOut != null) {
      args.add("--packagePath");
      args.add(apkOut.getExecPathString());
      outs.add(apkOut);
    }
    if (!resourceConfigs.isEmpty()) {
      args.add("--resourceConfigs");
      args.add(Joiner.on(',').join(resourceConfigs));
    }
    if (!densities.isEmpty()) {
      args.add("--densities");
      args.add(Joiner.on(',').join(densities));
    }
    if (!uncompressedExtensions.isEmpty()) {
      args.add("--uncompressedExtensions");
      args.add(Joiner.on(',').join(uncompressedExtensions));
    }
    if (!assetsToIgnore.isEmpty()) {
      args.add("--assetsToIgnore");
      args.add(
          Joiner.on(',').join(assetsToIgnore));
    }
    if (debug) {
      args.add("--debug");
    }

    if (versionCode != null) {
      args.add("--versionCode");
      args.add(versionCode);
    }

    if (versionName != null) {
      args.add("--versionName");
      args.add(versionName);
    }

    if (applicationId != null) {
      args.add("--applicationId");
      args.add(applicationId);
    }

    if (!Strings.isNullOrEmpty(customJavaPackage)) {
      // Sets an alternative java package for the generated R.java
      // this is allows android rules to generate resources outside of the java{,tests} tree.
      args.add("--packageForR");
      args.add(customJavaPackage);
    }

    // Create the spawn action.
    ruleContext.registerAction(this.spawnActionBuilder
        .addTool(sdk.getAapt())
        .addInputs(ImmutableList.<Artifact>copyOf(ins))
        .addOutputs(ImmutableList.<Artifact>copyOf(outs))
        .addArguments(ImmutableList.<String>copyOf(args))
        .setExecutable(
            ruleContext.getExecutablePrerequisite("$android_resources_processor", Mode.HOST))
        .setProgressMessage("Processing resources")
        .setMnemonic("AndroidAapt")
        .build(context));

    // Return the full set of processed transitive dependencies.
    return new ResourceContainer(
        primary.getLabel(),
        primary.getJavaPackage(),
        primary.getRenameManifestPackage(),
        primary.getConstantsInlined(),
        // If there is no apk to be generated, use the apk from the primary resources.
        // All ResourceContainers have to have an apk, but if a new one is not requested to be built
        // for this resource processing action (in case of just creating an R.txt or
        // proguard merging), reuse the primary resource from the dependencies.
        apkOut != null ? apkOut : primary.getApk(),
        primary.getManifest(),
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
