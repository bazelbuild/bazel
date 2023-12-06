// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.android.AndroidStarlarkData.fromNoneable;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDexInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** A provider for Android Dex artifacts */
@Immutable
public class AndroidDexInfo extends NativeInfo implements AndroidDexInfoApi<Artifact> {

  public static final String PROVIDER_NAME = "AndroidDexInfo";
  public static final Provider PROVIDER = new Provider();

  private final Artifact deployJar;
  private final Artifact finalClassesDexZip;
  private final Artifact filteredDeployJar;
  private final Artifact finalProguardOutputMap;
  private final Artifact javaResourceJar;
  private final Artifact shuffledJavaResourceJar;
  private final Artifact rexOutputPackageMap;

  public AndroidDexInfo(
      Artifact deployJar,
      Artifact finalClassesDexZip,
      Artifact filteredDeployJar,
      Artifact finalProguardOutputMap,
      Artifact javaResourceJar,
      Artifact shuffledJavaResourceJar,
      Artifact rexOutputPackageMap) {
    this.deployJar = deployJar;
    this.finalClassesDexZip = finalClassesDexZip;
    this.filteredDeployJar = filteredDeployJar;
    this.finalProguardOutputMap = finalProguardOutputMap;
    this.javaResourceJar = javaResourceJar;
    this.shuffledJavaResourceJar = shuffledJavaResourceJar;
    this.rexOutputPackageMap = rexOutputPackageMap;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  public Artifact getDeployJar() {
    return deployJar;
  }

  @Override
  @Nullable
  public Artifact getFinalClassesDexZip() {
    return finalClassesDexZip;
  }

  @Override
  @Nullable
  public Artifact getJavaResourceJar() {
    return javaResourceJar;
  }

  @Override
  @Nullable
  public Artifact getFinalProguardOutputMap() {
    return finalProguardOutputMap;
  }

  @Override
  @Nullable
  public Artifact getFilteredDeployJar() {
    return filteredDeployJar;
  }

  @Override
  @Nullable
  public Artifact getShuffledJavaResourceJar() {
    return shuffledJavaResourceJar;
  }

  @Override
  @Nullable
  public Artifact getRexOutputPackageMap() {
    return rexOutputPackageMap;
  }

  /** Provider for {@link AndroidDexInfo}. */
  public static class Provider extends BuiltinProvider<AndroidDexInfo>
      implements AndroidDexInfoApi.Provider<Artifact> {

    private Provider() {
      super(PROVIDER_NAME, AndroidDexInfo.class);
    }

    public String getName() {
      return PROVIDER_NAME;
    }

    @Override
    public AndroidDexInfo createInfo(
        Artifact deployJar,
        Object finalClassesDexZip,
        Object filteredDeployJar,
        Object finalProguardOutputMap,
        Object javaResourceJar,
        Object shuffledJavaResourceJar,
        Object rexOutputPackageMap)
        throws EvalException {

      return new AndroidDexInfo(
          deployJar,
          fromNoneable(finalClassesDexZip, Artifact.class),
          fromNoneable(filteredDeployJar, Artifact.class),
          fromNoneable(finalProguardOutputMap, Artifact.class),
          fromNoneable(javaResourceJar, Artifact.class),
          fromNoneable(shuffledJavaResourceJar, Artifact.class),
          fromNoneable(rexOutputPackageMap, Artifact.class));
    }
  }
}
