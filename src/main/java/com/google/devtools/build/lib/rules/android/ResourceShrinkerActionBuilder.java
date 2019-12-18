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
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import java.util.Collections;
import java.util.List;

/** Builder for creating resource shrinker actions. */
public class ResourceShrinkerActionBuilder {
  private AndroidAaptVersion targetAaptVersion;
  private Artifact resourceFilesZip;
  private Artifact shrunkJar;
  private Artifact proguardMapping;
  private ValidatedAndroidResources primaryResources;
  private ResourceDependencies dependencyResources;
  private Artifact resourceApkOut;
  private Artifact shrunkResourcesOut;
  private Artifact resourceOptimizationConfigOut;
  private Artifact logOut;

  private List<String> uncompressedExtensions = Collections.emptyList();
  private ResourceFilterFactory resourceFilterFactory = ResourceFilterFactory.empty();

  public ResourceShrinkerActionBuilder setUncompressedExtensions(
      List<String> uncompressedExtensions) {
    this.uncompressedExtensions = uncompressedExtensions;
    return this;
  }

  /** @param resourceFilterFactory The filters to apply to the resources. */
  public ResourceShrinkerActionBuilder setResourceFilterFactory(
      ResourceFilterFactory resourceFilterFactory) {
    this.resourceFilterFactory = resourceFilterFactory;
    return this;
  }

  /** @param resourceFilesZip A zip file containing the merged assets and resources to be shrunk. */
  public ResourceShrinkerActionBuilder withResourceFiles(Artifact resourceFilesZip) {
    this.resourceFilesZip = resourceFilesZip;
    return this;
  }

  /** @param shrunkJar The deploy jar of the rule after a dead code removal Proguard pass. */
  public ResourceShrinkerActionBuilder withShrunkJar(Artifact shrunkJar) {
    this.shrunkJar = shrunkJar;
    return this;
  }

  /** @param proguardMapping The Proguard mapping between obfuscated and original code. */
  public ResourceShrinkerActionBuilder withProguardMapping(Artifact proguardMapping) {
    this.proguardMapping = proguardMapping;
    return this;
  }

  /**
   * @param primary The fully processed {@link ValidatedAndroidResources} of the resources to be
   *     shrunk. Must contain both an R.txt and merged manifest.
   */
  public ResourceShrinkerActionBuilder withPrimary(ValidatedAndroidResources primary) {
    checkNotNull(primary);
    checkNotNull(primary.getManifest());
    checkNotNull(primary.getRTxt());
    this.primaryResources = primary;
    return this;
  }

  /** @param resourceDeps The full dependency tree of resources. */
  public ResourceShrinkerActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencyResources = resourceDeps;
    return this;
  }

  /** @param resourceApkOut The location to write the shrunk resource ap_ package. */
  public ResourceShrinkerActionBuilder setResourceApkOut(Artifact resourceApkOut) {
    this.resourceApkOut = resourceApkOut;
    return this;
  }

  /** @param shrunkResourcesOut The location to write the shrunk resource files zip. */
  public ResourceShrinkerActionBuilder setShrunkResourcesOut(Artifact shrunkResourcesOut) {
    this.shrunkResourcesOut = shrunkResourcesOut;
    return this;
  }

  /** @param resourceOptimizationConfigOut The location to write the config for the optimizer. */
  public ResourceShrinkerActionBuilder setResourceOptimizationConfigOut(
      Artifact resourceOptimizationConfigOut) {
    this.resourceOptimizationConfigOut = resourceOptimizationConfigOut;
    return this;
  }

  /** @param logOut The location to write the shrinker log. */
  public ResourceShrinkerActionBuilder setLogOut(Artifact logOut) {
    this.logOut = logOut;
    return this;
  }

  /** @param androidAaptVersion The aapt version to target with this action. */
  public ResourceShrinkerActionBuilder setTargetAaptVersion(AndroidAaptVersion androidAaptVersion) {
    this.targetAaptVersion = androidAaptVersion;
    return this;
  }

  public Artifact build(AndroidDataContext dataContext) {

    checkNotNull(resourceFilesZip);
    checkNotNull(shrunkJar);
    checkNotNull(proguardMapping);
    checkNotNull(primaryResources);
    checkNotNull(primaryResources.getRTxt());
    checkNotNull(primaryResources.getManifest());
    checkNotNull(resourceApkOut);

    BusyBoxActionBuilder builder;
    if (targetAaptVersion == AndroidAaptVersion.AAPT2) {
      builder = BusyBoxActionBuilder.create(dataContext, "SHRINK_AAPT2");
    } else {
      builder =
          BusyBoxActionBuilder.create(dataContext, "SHRINK")
              .maybeAddVectoredFlag("--uncompressedExtensions", uncompressedExtensions)
              // Order, for some reason, is important.
              .addVectoredFlag(
                  "--resourcePackages", getResourcePackages(primaryResources, dependencyResources))
              .addInput("--primaryManifest", primaryResources.getManifest())
              .maybeAddInput("--dependencyManifest", getManifests(dependencyResources))
              .maybeAddFlag(
                  "--resourceConfigs", resourceFilterFactory.getConfigurationFilterString());
    }

    builder
        .addAapt(targetAaptVersion)
        .addAndroidJar()
        .maybeAddFlag("--debug", dataContext.useDebug())
        .addInput("--resources", resourceFilesZip)
        .addInput("--shrunkJar", shrunkJar)
        .addInput("--proguardMapping", proguardMapping)
        .addInput("--rTxt", primaryResources.getRTxt())
        .addOutput("--shrunkResourceApk", resourceApkOut)
        .addOutput("--shrunkResources", shrunkResourcesOut)
        .maybeAddOutput("--resourcesConfigOutput", resourceOptimizationConfigOut)
        .addOutput("--log", logOut)
        .buildAndRegister("Shrinking resources", "ResourceShrinker");

    return resourceApkOut;
  }

  private ImmutableList<Artifact> getManifests(ResourceDependencies resourceDependencies) {
    ImmutableList.Builder<Artifact> manifests = ImmutableList.builder();
    for (ValidatedAndroidResources resources :
        resourceDependencies.getResourceContainers().toList()) {
      if (resources.getManifest() != null) {
        manifests.add(resources.getManifest());
      }
    }
    return manifests.build();
  }

  private ImmutableList<String> getResourcePackages(
      ValidatedAndroidResources primaryResources, ResourceDependencies resourceDependencies) {
    ImmutableList.Builder<String> resourcePackages = ImmutableList.builder();
    resourcePackages.add(primaryResources.getJavaPackage());
    for (ValidatedAndroidResources resources :
        resourceDependencies.getResourceContainers().toList()) {
      resourcePackages.add(resources.getJavaPackage());
    }
    return resourcePackages.build();
  }
}
