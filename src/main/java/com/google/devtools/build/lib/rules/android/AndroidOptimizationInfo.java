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
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidOptimizationInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Provider for proguard + resource shrinking artifacts. */
@Immutable
public class AndroidOptimizationInfo extends NativeInfo
    implements AndroidOptimizationInfoApi<Artifact> {

  public static final String PROVIDER_NAME = "AndroidOptimizationInfo";
  public static final Provider PROVIDER = new Provider();

  @Nullable private final Artifact optimizedJar;
  @Nullable private final Artifact mapping;
  @Nullable private final Artifact seeds;
  @Nullable private final Artifact libraryJar;
  @Nullable private final Artifact config;
  @Nullable private final Artifact usage;
  @Nullable private final Artifact protoMapping;
  @Nullable private final Artifact rewrittenStartupProfile;
  @Nullable private final Artifact rewrittenMergedBaselineProfile;
  @Nullable private final Artifact optimizedResourceApk;
  @Nullable private final Artifact shrunkResourceApk;
  @Nullable private final Artifact shrunkResourceZip;
  @Nullable private final Artifact resourceShrinkerLog;
  @Nullable private final Artifact resourceOptimizationConfig;
  @Nullable private final Artifact resourcePathShorteningMap;

  public AndroidOptimizationInfo(
      Artifact optimizedJar,
      Artifact mapping,
      Artifact seeds,
      Artifact libraryJar,
      Artifact config,
      Artifact usage,
      Artifact protoMapping,
      Artifact rewrittenStartupProfile,
      Artifact rewrittenMergedBaselineProfile,
      Artifact optimizedResourceApk,
      Artifact shrunkResourceApk,
      Artifact shrunkResourceZip,
      Artifact resourceShrinkerLog,
      Artifact resourceOptimizationConfig,
      Artifact resourcePathShorteningMap) {
    this.optimizedJar = optimizedJar;
    this.mapping = mapping;
    this.seeds = seeds;
    this.libraryJar = libraryJar;
    this.config = config;
    this.usage = usage;
    this.protoMapping = protoMapping;
    this.rewrittenStartupProfile = rewrittenStartupProfile;
    this.rewrittenMergedBaselineProfile = rewrittenMergedBaselineProfile;
    this.optimizedResourceApk = optimizedResourceApk;
    this.shrunkResourceApk = shrunkResourceApk;
    this.shrunkResourceZip = shrunkResourceZip;
    this.resourceShrinkerLog = resourceShrinkerLog;
    this.resourceOptimizationConfig = resourceOptimizationConfig;
    this.resourcePathShorteningMap = resourcePathShorteningMap;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  @Override
  @Nullable
  public Artifact getOptimizedJar() {
    return optimizedJar;
  }

  @Override
  @Nullable
  public Artifact getMapping() {
    return mapping;
  }

  @Override
  @Nullable
  public Artifact getSeeds() {
    return seeds;
  }

  @Override
  @Nullable
  public Artifact getLibraryJar() {
    return libraryJar;
  }

  @Override
  @Nullable
  public Artifact getConfig() {
    return config;
  }

  @Override
  @Nullable
  public Artifact getUsage() {
    return usage;
  }

  @Override
  @Nullable
  public Artifact getProtoMapping() {
    return protoMapping;
  }

  @Override
  @Nullable
  public Artifact getRewrittenStartupProfile() {
    return rewrittenStartupProfile;
  }

  @Override
  @Nullable
  public Artifact getRewrittenMergedBaselineProfile() {
    return rewrittenMergedBaselineProfile;
  }

  @Override
  @Nullable
  public Artifact getOptimizedResourceApk() {
    return optimizedResourceApk;
  }

  @Override
  @Nullable
  public Artifact getShrunkResourceApk() {
    return shrunkResourceApk;
  }

  @Override
  @Nullable
  public Artifact getShrunkResourceZip() {
    return shrunkResourceZip;
  }

  @Override
  @Nullable
  public Artifact getResourceShrinkerLog() {
    return resourceShrinkerLog;
  }

  @Override
  @Nullable
  public Artifact getResourceOptimizationConfig() {
    return resourceOptimizationConfig;
  }

  @Override
  @Nullable
  public Artifact getResourcePathShorteningMap() {
    return resourcePathShorteningMap;
  }

  /** Provider for {@link AndroidOptimizationInfoApi}. */
  public static class Provider extends BuiltinProvider<AndroidOptimizationInfo>
      implements AndroidOptimizationInfoApi.Provider<Artifact> {

    private Provider() {
      super(PROVIDER_NAME, AndroidOptimizationInfo.class);
    }

    public String getName() {
      return PROVIDER_NAME;
    }

    @Override
    public AndroidOptimizationInfo createInfo(
        Object optimizedJar,
        Object mapping,
        Object seeds,
        Object libraryJar,
        Object config,
        Object usage,
        Object protoMapping,
        Object rewrittenStartupProfile,
        Object rewrittenMergedBaselineProfile,
        Object optimizedResourceApk,
        Object shrunkResourceApk,
        Object shrunkResourceZip,
        Object resourceShrinkerLog,
        Object resourceOptimizationConfig,
        Object resourcePathShorteningMap)
        throws EvalException {
      return new AndroidOptimizationInfo(
          fromNoneable(optimizedJar, Artifact.class),
          fromNoneable(mapping, Artifact.class),
          fromNoneable(seeds, Artifact.class),
          fromNoneable(libraryJar, Artifact.class),
          fromNoneable(config, Artifact.class),
          fromNoneable(usage, Artifact.class),
          fromNoneable(protoMapping, Artifact.class),
          fromNoneable(rewrittenStartupProfile, Artifact.class),
          fromNoneable(rewrittenMergedBaselineProfile, Artifact.class),
          fromNoneable(optimizedResourceApk, Artifact.class),
          fromNoneable(shrunkResourceApk, Artifact.class),
          fromNoneable(shrunkResourceZip, Artifact.class),
          fromNoneable(resourceShrinkerLog, Artifact.class),
          fromNoneable(resourceOptimizationConfig, Artifact.class),
          fromNoneable(resourcePathShorteningMap, Artifact.class));
    }
  }
}
