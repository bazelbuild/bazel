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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.AndroidAaptVersion;
import com.google.devtools.build.lib.rules.android.AndroidDataConverter.JoinerType;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.function.Function;

/** Builds up the spawn action for $android_rclass_generator. */
public class RClassGeneratorActionBuilder {

  @AutoCodec @VisibleForSerialization
  static final AndroidDataConverter<ValidatedAndroidResources> AAPT2_CONVERTER =
      AndroidDataConverter.<ValidatedAndroidResources>builder(JoinerType.COLON_COMMA)
          .with(chooseDepsToArg(AndroidAaptVersion.AAPT2))
          .build();

  private ResourceDependencies dependencies;

  private Artifact classJarOut;

  private boolean finalFields = true;

  public RClassGeneratorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.dependencies = resourceDeps;
    return this;
  }

  public RClassGeneratorActionBuilder finalFields(boolean finalFields) {
    this.finalFields = finalFields;
    return this;
  }

  public RClassGeneratorActionBuilder setClassJarOut(Artifact classJarOut) {
    this.classJarOut = classJarOut;
    return this;
  }

  public ResourceApk build(AndroidDataContext dataContext, ProcessedAndroidData data) {
    build(dataContext, data.getRTxt(), data.getManifest());

    return data.withValidatedResources(classJarOut);
  }

  private void build(
      AndroidDataContext dataContext, Artifact rTxt, ProcessedAndroidManifest manifest) {
    BusyBoxActionBuilder builder =
        BusyBoxActionBuilder.create(dataContext, "GENERATE_BINARY_R")
            .addInput("--primaryRTxt", rTxt)
            .addInput("--primaryManifest", manifest.getManifest())
            .maybeAddFlag("--packageForR", manifest.getPackage())
            .addFlag(finalFields ? "--finalFields" : "--nofinalFields");

    if (dependencies != null && !dependencies.getResourceContainers().isEmpty()) {
      builder
          .addTransitiveFlagForEach(
              "--library", dependencies.getResourceContainers(), AAPT2_CONVERTER)
          .addTransitiveInputValues(dependencies.getTransitiveAapt2RTxt())
          .addTransitiveInputValues(dependencies.getTransitiveManifests())
          .addTransitiveInputValues(dependencies.getTransitiveAapt2ValidationArtifacts());
    }

    builder
        .addOutput("--classJarOutput", classJarOut)
        .addLabelFlag("--targetLabel")
        .buildAndRegister("Generating R Classes", "RClassGenerator");
  }

  private static Function<ValidatedAndroidResources, String> chooseDepsToArg(
      final AndroidAaptVersion version) {
    return container -> {
      Artifact rTxt =
          version == AndroidAaptVersion.AAPT2 ? container.getAapt2RTxt() : container.getRTxt();
      return (rTxt != null ? rTxt.getExecPath() : "")
          + ","
          + (container.getManifest() != null ? container.getManifest().getExecPath() : "");
    };
  }
}
