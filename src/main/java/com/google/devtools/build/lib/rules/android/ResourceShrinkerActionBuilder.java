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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import java.util.Collections;
import java.util.List;

/**
 * Builder for creating resource shrinker actions.
 */
public class ResourceShrinkerActionBuilder {
  private Artifact resourceFilesZip;
  private Artifact shrunkJar;
  private Artifact proguardMapping;
  private ResourceContainer primaryResources;
  private ResourceDependencies dependencyResources;
  private Artifact resourceApkOut;
  private Artifact shrunkResourcesOut;
  private Artifact logOut;

  private final RuleContext ruleContext;
  private final SpawnAction.Builder spawnActionBuilder;
  private final AndroidSdkProvider sdk;

  private List<String> uncompressedExtensions = Collections.emptyList();
  private List<String> assetsToIgnore = Collections.emptyList();
  private ResourceFilter resourceFilter;

  /**
   * @param ruleContext The RuleContext of the owning rule.
   */
  public ResourceShrinkerActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
    this.spawnActionBuilder = new SpawnAction.Builder();
    this.sdk = AndroidSdkProvider.fromRuleContext(ruleContext);
    this.resourceFilter = ResourceFilter.empty(ruleContext);
  }

  public ResourceShrinkerActionBuilder setUncompressedExtensions(
      List<String> uncompressedExtensions) {
    this.uncompressedExtensions = uncompressedExtensions;
    return this;
  }

  public ResourceShrinkerActionBuilder setAssetsToIgnore(List<String> assetsToIgnore) {
    this.assetsToIgnore = assetsToIgnore;
    return this;
  }

  /** @param resourceFilter The filters to apply to the resources. */
  public ResourceShrinkerActionBuilder setResourceFilter(ResourceFilter resourceFilter) {
    this.resourceFilter = resourceFilter;
    return this;
  }

  /**
   * @param resourceFilesZip A zip file containing the merged assets and resources to be shrunk.
   */
  public ResourceShrinkerActionBuilder withResourceFiles(Artifact resourceFilesZip) {
    this.resourceFilesZip = resourceFilesZip;
    return this;
  }

  /**
   * @param shrunkJar The deploy jar of the rule after a dead code removal Proguard pass.
   */
  public ResourceShrinkerActionBuilder withShrunkJar(Artifact shrunkJar) {
    this.shrunkJar = shrunkJar;
    return this;
  }

  /**
   * @param proguardMapping The Proguard mapping between obfuscated and original code.
   */
  public ResourceShrinkerActionBuilder withProguardMapping(Artifact proguardMapping) {
    this.proguardMapping = proguardMapping;
    return this;
  }

  /**
   * @param primary The fully processed {@link ResourceContainer} of the resources to be shrunk.
   *     Must contain both an R.txt and merged manifest.
   */
  public ResourceShrinkerActionBuilder withPrimary(ResourceContainer primary) {
    checkNotNull(primary);
    checkNotNull(primary.getManifest());
    checkNotNull(primary.getRTxt());
    this.primaryResources = primary;
    return this;
  }

  /**
   * @param resourceDeps The full dependency tree of {@link ResourceContainer}s.
   */
  public ResourceShrinkerActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencyResources = resourceDeps;
    return this;
  }

  /**
   * @param resourceApkOut The location to write the shrunk resource ap_ package.
   */
  public ResourceShrinkerActionBuilder setResourceApkOut(Artifact resourceApkOut) {
    this.resourceApkOut = resourceApkOut;
    return this;
  }

  /**
   * @param shrunkResourcesOut The location to write the shrunk resource files zip.
   */
  public ResourceShrinkerActionBuilder setShrunkResourcesOut(Artifact shrunkResourcesOut) {
    this.shrunkResourcesOut = shrunkResourcesOut;
    return this;
  }

  /**
   * @param logOut The location to write the shrinker log.
   */
  public ResourceShrinkerActionBuilder setLogOut(Artifact logOut) {
    this.logOut = logOut;
    return this;
  }

  public Artifact build() throws RuleErrorException {
    if (AndroidAaptVersion.chooseTargetAaptVersion(ruleContext) == AndroidAaptVersion.AAPT2) {
      ruleContext.throwWithRuleError("aapt2 enabled builds do not yet support resource shrinking.");
    }
    ImmutableList.Builder<Artifact> inputs = ImmutableList.builder();
    ImmutableList.Builder<Artifact> outputs = ImmutableList.builder();

    CustomCommandLine.Builder commandLine = new CustomCommandLine.Builder();

    // Set the busybox tool.
    commandLine.add("--tool").add("SHRINK").add("--");

    inputs.addAll(
        ruleContext
            .getExecutablePrerequisite("$android_resources_busybox", Mode.HOST)
            .getRunfilesSupport()
            .getRunfilesArtifactsWithoutMiddlemen());

    commandLine.addExecPath("--aapt", sdk.getAapt().getExecutable());

    commandLine.addExecPath("--annotationJar", sdk.getAnnotationsJar());
    inputs.add(sdk.getAnnotationsJar());

    commandLine.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    if (!uncompressedExtensions.isEmpty()) {
      commandLine.addJoined("--uncompressedExtensions", ",", uncompressedExtensions);
    }
    if (!assetsToIgnore.isEmpty()) {
      commandLine.addJoined("--assetsToIgnore", ",", assetsToIgnore);
    }
    if (ruleContext.getConfiguration().getCompilationMode() != CompilationMode.OPT) {
      commandLine.add("--debug");
    }
    if (resourceFilter.hasConfigurationFilters()) {
      commandLine.add("--resourceConfigs", resourceFilter.getConfigurationFilterString());
    }

    checkNotNull(resourceFilesZip);
    checkNotNull(shrunkJar);
    checkNotNull(proguardMapping);
    checkNotNull(primaryResources);
    checkNotNull(primaryResources.getRTxt());
    checkNotNull(primaryResources.getManifest());
    checkNotNull(resourceApkOut);

    commandLine.addExecPath("--resources", resourceFilesZip);
    inputs.add(resourceFilesZip);

    commandLine.addExecPath("--shrunkJar", shrunkJar);
    inputs.add(shrunkJar);

    commandLine.addExecPath("--proguardMapping", proguardMapping);
    inputs.add(proguardMapping);

    commandLine.addExecPath("--rTxt", primaryResources.getRTxt());
    inputs.add(primaryResources.getRTxt());

    commandLine.addExecPath("--primaryManifest", primaryResources.getManifest());
    inputs.add(primaryResources.getManifest());

    ImmutableList<Artifact> dependencyManifests = getManifests(dependencyResources);
    if (!dependencyManifests.isEmpty()) {
      commandLine.addExecPaths("--dependencyManifest", dependencyManifests);
      inputs.addAll(dependencyManifests);
    }

    ImmutableList<String> resourcePackages =
        getResourcePackages(primaryResources, dependencyResources);
    commandLine.addJoined("--resourcePackages", ",", resourcePackages);

    commandLine.addExecPath("--shrunkResourceApk", resourceApkOut);
    outputs.add(resourceApkOut);

    commandLine.addExecPath("--shrunkResources", shrunkResourcesOut);
    outputs.add(shrunkResourcesOut);

    commandLine.addExecPath("--log", logOut);
    outputs.add(logOut);

    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTool(sdk.getAapt())
            .addInputs(inputs.build())
            .addOutputs(outputs.build())
            .setCommandLine(commandLine.build())
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("Shrinking resources for %s", ruleContext.getLabel())
            .setMnemonic("ResourceShrinker")
            .build(ruleContext));

    return resourceApkOut;
  }

  private ImmutableList<Artifact> getManifests(ResourceDependencies resourceDependencies) {
    ImmutableList.Builder<Artifact> manifests = ImmutableList.builder();
    for (ResourceContainer resources : resourceDependencies.getResources()) {
      if (resources.getManifest() != null) {
        manifests.add(resources.getManifest());
      }
    }
    return manifests.build();
  }

  private ImmutableList<String> getResourcePackages(
      ResourceContainer primaryResources, ResourceDependencies resourceDependencies) {
    ImmutableList.Builder<String> resourcePackages = ImmutableList.builder();
    resourcePackages.add(primaryResources.getJavaPackage());
    for (ResourceContainer resources : resourceDependencies.getResources()) {
      resourcePackages.add(resources.getJavaPackage());
    }
    return resourcePackages.build();
  }
}

