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

import com.google.devtools.build.lib.actions.Artifact;
import java.util.Collections;
import java.util.List;

/** Builder for creating resource shrinker actions. */
public class ResourceShrinkerActionBuilder {
  private Artifact resourceFilesZip;
  private Artifact shrunkJar;
  private Artifact proguardMapping;
  private Artifact rTxt;
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

  /** @param rTxt The R.txt file produced during resource packaging. */
  public ResourceShrinkerActionBuilder withRTxt(Artifact rTxt) {
    this.rTxt = rTxt;
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

  public Artifact build(AndroidDataContext dataContext) {

    checkNotNull(resourceFilesZip);
    checkNotNull(shrunkJar);
    checkNotNull(proguardMapping);
    checkNotNull(rTxt);
    checkNotNull(resourceApkOut);

    BusyBoxActionBuilder builder = BusyBoxActionBuilder.create(dataContext, "SHRINK_AAPT2");

    builder
        .addAapt()
        .addAndroidJar()
        .maybeAddFlag("--debug", dataContext.useDebug())
        .addInput("--resources", resourceFilesZip)
        .addInput("--shrunkJar", shrunkJar)
        .addInput("--proguardMapping", proguardMapping)
        .addInput("--rTxt", rTxt)
        .addOutput("--shrunkResourceApk", resourceApkOut)
        .addOutput("--shrunkResources", shrunkResourcesOut)
        .maybeAddOutput("--resourcesConfigOutput", resourceOptimizationConfigOut)
        .addOutput("--log", logOut)
        .buildAndRegister("Shrinking resources", "ResourceShrinker");

    return resourceApkOut;
  }
}
