// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.AndroidDataConverter.JoinerType;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;

/**
 * Builder for generating R classes for robolectric action.
 *
 * <p>This builder will create an action that generates r classes with internally consistent
 * resource ids for the transitive closure of dependencies that provide resources.
 */
public class RobolectricResourceSymbolsActionBuilder {

  @AutoCodec @VisibleForSerialization
  static final AndroidDataConverter<ValidatedAndroidData> TO_ARG =
      AndroidDataConverter.<ValidatedAndroidData>builder(JoinerType.COLON_COMMA)
          .withRoots(ValidatedAndroidData::getResourceRoots)
          .withRoots(ValidatedAndroidData::getAssetRoots)
          .withArtifact(ValidatedAndroidData::getManifest)
          .maybeWithArtifact(ValidatedAndroidData::getRTxt)
          .maybeWithArtifact(ValidatedAndroidData::getSymbols)
          .build();

  @AutoCodec @VisibleForSerialization
  static final AndroidDataConverter<ValidatedAndroidData> TO_ARG_AAPT2 =
      AndroidDataConverter.<ValidatedAndroidData>builder(JoinerType.COLON_COMMA)
          .withRoots(ValidatedAndroidData::getResourceRoots)
          .withRoots(ValidatedAndroidData::getAssetRoots)
          .withArtifact(ValidatedAndroidData::getManifest)
          .maybeWithArtifact(ValidatedAndroidData::getAapt2RTxt)
          .maybeWithArtifact(ValidatedAndroidData::getSymbols)
          .build();

  private Artifact classJarOut;
  private final ResourceDependencies dependencies;
  private AndroidAaptVersion androidAaptVersion;

  private RobolectricResourceSymbolsActionBuilder(ResourceDependencies dependencies) {
    this.dependencies = dependencies;
  }

  public static RobolectricResourceSymbolsActionBuilder create(ResourceDependencies dependencies) {
    return new RobolectricResourceSymbolsActionBuilder(dependencies);
  }

  public RobolectricResourceSymbolsActionBuilder setJarOut(Artifact classJar) {
    this.classJarOut = classJar;
    return this;
  }

  public RobolectricResourceSymbolsActionBuilder targetAaptVersion(
      AndroidAaptVersion androidAaptVersion) {
    this.androidAaptVersion = androidAaptVersion;
    return this;
  }

  public NestedSet<Artifact> buildAsClassPathEntry(AndroidDataContext dataContext) {
    BusyBoxActionBuilder builder =
        BusyBoxActionBuilder.create(dataContext, "GENERATE_ROBOLECTRIC_R").addAndroidJar();

    if (!dependencies.getResourceContainers().isEmpty()) {
      builder
          .addTransitiveFlag(
              "--data",
              dependencies.getResourceContainers(),
              androidAaptVersion == AndroidAaptVersion.AAPT2 ? TO_ARG_AAPT2 : TO_ARG)
          .addTransitiveInputValues(
              androidAaptVersion == AndroidAaptVersion.AAPT2
                  ? dependencies.getTransitiveAapt2RTxt()
                  : dependencies.getTransitiveRTxt())
          .addTransitiveInputValues(dependencies.getTransitiveResources())
          .addTransitiveInputValues(dependencies.getTransitiveAssets())
          .addTransitiveInputValues(dependencies.getTransitiveManifests())
          .addTransitiveInputValues(dependencies.getTransitiveSymbolsBin());
    }

    builder
        .addOutput("--classJarOutput", classJarOut)
        .addLabelFlag("--targetLabel")
        .buildAndRegister("Generating R classes", "GenerateRobolectricRClasses");

    return NestedSetBuilder.<Artifact>naiveLinkOrder().add(classJarOut).build();
  }
}
