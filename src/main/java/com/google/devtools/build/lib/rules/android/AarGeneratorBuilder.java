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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;

/** Builder for creating aar generator action. */
public class AarGeneratorBuilder {

  private AndroidResources primaryResources;
  private AndroidAssets primaryAssets;

  private Artifact manifest;
  private Artifact rTxt;
  private Artifact classes;
  private ImmutableList<Artifact> proguardSpecs = ImmutableList.of();

  private Artifact aarOut;
  private boolean throwOnResourceConflict;

  public AarGeneratorBuilder withPrimaryResources(AndroidResources primaryResources) {
    this.primaryResources = primaryResources;
    return this;
  }

  public AarGeneratorBuilder withPrimaryAssets(AndroidAssets primaryAssets) {
    this.primaryAssets = primaryAssets;
    return this;
  }

  public AarGeneratorBuilder withManifest(Artifact manifest) {
    this.manifest = manifest;
    return this;
  }

  public AarGeneratorBuilder withRtxt(Artifact rTxt) {
    this.rTxt = rTxt;
    return this;
  }

  public AarGeneratorBuilder withClasses(Artifact classes) {
    this.classes = classes;
    return this;
  }

  public AarGeneratorBuilder setAAROut(Artifact aarOut) {
    this.aarOut = aarOut;
    return this;
  }

  public AarGeneratorBuilder setProguardSpecs(ImmutableList<Artifact> proguardSpecs) {
    this.proguardSpecs = proguardSpecs;
    return this;
  }

  public AarGeneratorBuilder setThrowOnResourceConflict(boolean throwOnResourceConflict) {
    this.throwOnResourceConflict = throwOnResourceConflict;
    return this;
  }

  public void build(AndroidDataContext dataContext) {
    BusyBoxActionBuilder builder =
        BusyBoxActionBuilder.create(dataContext, "GENERATE_AAR")
            // no R.txt, because it will be generated from this action.
            .addInput(
                "--mainData",
                String.format(
                    "%s:%s:%s",
                    AndroidDataConverter.rootsToString(primaryResources.getResourceRoots()),
                    AndroidDataConverter.rootsToString(primaryAssets.getAssetRoots()),
                    manifest.getExecPathString()),
                Iterables.concat(
                    primaryResources.getResources(),
                    primaryAssets.getAssets(),
                    ImmutableList.of(manifest)))
            .addInput("--manifest", manifest)
            .maybeAddInput("--rtxt", rTxt)
            .maybeAddInput("--classes", classes);

    for (Artifact proguardSpec : proguardSpecs) {
      builder.addInput("--proguardSpec", proguardSpec);
    }

    builder
        .addOutput("--aarOutput", aarOut)
        .maybeAddFlag("--throwOnResourceConflict", throwOnResourceConflict)
        .buildAndRegister("Building AAR package", "AARGenerator");
  }
}
