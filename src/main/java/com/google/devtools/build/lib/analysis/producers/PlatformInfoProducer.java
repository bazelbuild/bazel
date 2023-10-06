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
package com.google.devtools.build.lib.analysis.producers;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import javax.annotation.Nullable;

/**
 * Retrieves {@link PlatformInfo} for a given platform.
 *
 * <p>Since platforms do not rely on the configuration, this uses a dummy blank configuration to
 * help reduce the number of skyframe edges created.
 *
 * <p>This creates an explicit dependency on the {@link Package} to retrieve the associated target,
 * so it is possible to verify that {@link PlatformInfo} is an advertised provider before
 * constructing the {@link ConfiguredTarget}.
 */
final class PlatformInfoProducer
    implements StateMachine, StateMachine.ValueOrExceptionSink<NoSuchPackageException> {
  interface ResultSink {
    void acceptPlatformInfo(PlatformInfo info);

    void acceptPlatformInfoError(InvalidPlatformException error);
  }

  // -------------------- Input --------------------
  private final Label platformLabel;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  private boolean passedValidation = false;
  private ConfiguredTarget platform;

  PlatformInfoProducer(Label platformLabel, ResultSink sink, StateMachine runAfter) {
    this.platformLabel = platformLabel;
    this.sink = sink;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    // Loads the Package first to verify the Target. The ConfiguredTarget should not be loaded
    // until after verification. See https://github.com/bazelbuild/bazel/pull/10307.
    //
    // In distributed analysis, these packages will be duplicated across shards.
    tasks.lookUp(
        platformLabel.getPackageIdentifier(),
        NoSuchPackageException.class,
        (StateMachine.ValueOrExceptionSink<NoSuchPackageException>) this);
    return this::lookupPlatform;
  }

  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable NoSuchPackageException error) {
    if (value != null) {
      var pkg = ((PackageValue) value).getPackage();
      try {
        var target = pkg.getTarget(platformLabel.getName());
        if (!PlatformLookupUtil.hasPlatformInfo(target)) {
          // validation failure
          sink.acceptPlatformInfoError(new InvalidPlatformException(platformLabel));
          return;
        }
      } catch (NoSuchTargetException e) {
        sink.acceptPlatformInfoError(new InvalidPlatformException(e));
        return;
      }
      passedValidation = true;
      return;
    }
    if (error != null) {
      sink.acceptPlatformInfoError(new InvalidPlatformException(error));
      return;
    }
    throw new IllegalArgumentException("both value and error were null");
  }

  private StateMachine lookupPlatform(Tasks tasks) {
    if (!passedValidation) {
      return runAfter;
    }

    // Create a configured target key with a dummy configuration.
    ConfiguredTargetKey platformKey =
        ConfiguredTargetKey.builder()
            .setLabel(platformLabel)
            .setConfigurationKey(BuildConfigurationKey.create(CommonOptions.EMPTY_OPTIONS))
            .build();
    tasks.lookUp(
        platformKey, ConfiguredValueCreationException.class, this::acceptPlatformValueOrError);
    return this::retrievePlatformInfo;
  }

  private void acceptPlatformValueOrError(
      @Nullable SkyValue value, @Nullable ConfiguredValueCreationException error) {
    if (value != null) {
      var configuredTargetValue = (ConfiguredTargetValue) value;
      this.platform = configuredTargetValue.getConfiguredTarget();
      return;
    }
    if (error != null) {
      sink.acceptPlatformInfoError(new InvalidPlatformException(platformLabel, error));
      return;
    }
    throw new IllegalArgumentException("both value and error were null");
  }

  private StateMachine retrievePlatformInfo(Tasks tasks) {
    if (platform == null) {
      return runAfter; // An error occurred and was reported.
    }

    PlatformInfo platformInfo = PlatformProviderUtils.platform(platform);
    if (platformInfo == null) {
      sink.acceptPlatformInfoError(new InvalidPlatformException(platform.getLabel()));
      return runAfter;
    }

    sink.acceptPlatformInfo(platformInfo);
    return runAfter;
  }
}
