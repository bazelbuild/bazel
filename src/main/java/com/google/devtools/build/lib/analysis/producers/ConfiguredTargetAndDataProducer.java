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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Determines {@link ConfiguredTargetAndData} from {@link ConfiguredTargetKey}.
 *
 * <p>The resulting package and configuration are based on the resulting {@link ConfiguredTarget}
 * and may be different from what is in the key, for example, if there is an alias.
 */
final class ConfiguredTargetAndDataProducer
    implements StateMachine,
        Consumer<SkyValue>,
        StateMachine.ValueOrExceptionSink<ConfiguredValueCreationException> {
  /** Interface for accepting values produced by this class. */
  interface ResultSink {
    void acceptConfiguredTargetAndData(ConfiguredTargetAndData value, int index);

    void acceptConfiguredTargetAndDataError(ConfiguredValueCreationException error);
  }

  // -------------------- Input --------------------
  private final ConfiguredTargetKey key;
  @Nullable // Null if no transition key is needed (patch transition or no-op split transition).
  private final String transitionKey;
  private final TransitiveDependencyState transitiveState;

  // -------------------- Output --------------------
  private final ResultSink sink;
  private final int outputIndex;

  // -------------------- Internal State --------------------
  private ConfiguredTarget configuredTarget;
  @Nullable // Null if the configured target key's configuration key is null.
  private BuildConfigurationValue configurationValue;
  private Package pkg;

  public ConfiguredTargetAndDataProducer(
      ConfiguredTargetKey key,
      @Nullable String transitionKey,
      TransitiveDependencyState transitiveState,
      ResultSink sink,
      int outputIndex) {
    this.key = key;
    this.transitionKey = transitionKey;
    this.transitiveState = transitiveState;
    this.sink = sink;
    this.outputIndex = outputIndex;
  }

  @Override
  public StateMachine step(Tasks tasks, ExtendedEventHandler listener) {
    tasks.lookUp(
        key.toKey(),
        ConfiguredValueCreationException.class,
        (ValueOrExceptionSink<ConfiguredValueCreationException>) this);
    return this::fetchConfigurationAndPackage;
  }

  @Override
  public void acceptValueOrException(
      @Nullable SkyValue value, @Nullable ConfiguredValueCreationException error) {
    if (value != null) {
      var configuredTargetValue = (ConfiguredTargetValue) value;
      this.configuredTarget = configuredTargetValue.getConfiguredTarget();
      transitiveState.updateTransitivePackages(configuredTargetValue);
      return;
    }
    if (error != null) {
      transitiveState.addTransitiveCauses(error.getRootCauses());
      sink.acceptConfiguredTargetAndDataError(error);
      return;
    }
    throw new IllegalArgumentException("both value and error were null");
  }

  private StateMachine fetchConfigurationAndPackage(Tasks tasks, ExtendedEventHandler listener) {
    if (configuredTarget == null) {
      return DONE; // There was a previous error.
    }

    var configurationKey = configuredTarget.getConfigurationKey();
    if (configurationKey != null) {
      tasks.lookUp(configurationKey, (Consumer<SkyValue>) this);
    }

    // An alternative to this is to optimistically fetch the package using the label of the
    // configured target key. However, the actual package may differ when this is an
    // AliasConfiguredTarget and would need to be refetched.

    // TODO(shahan): This lookup should be skipped when the ConfiguredTarget is fetched remotely.
    var packageId = configuredTarget.getLabel().getPackageIdentifier();
    this.pkg = transitiveState.getDependencyPackage(packageId);
    if (pkg == null) {
      // In incremental builds, it is possible that the package won't be present in the cache. For
      // example, suppose that a configured target A has two children B and C. If B is dirty, it
      // causes A's re-evaluation, which causes this fetch to be performed for C. However, C has not
      // been evaluated this build.
      tasks.lookUp(packageId, (Consumer<SkyValue>) this);
    }

    return this::constructResult;
  }

  @Override
  public void accept(SkyValue value) {
    if (value instanceof BuildConfigurationValue) {
      this.configurationValue = (BuildConfigurationValue) value;
      return;
    }
    if (value instanceof PackageValue) {
      this.pkg = ((PackageValue) value).getPackage();
      return;
    }
    throw new IllegalArgumentException("unexpected value: " + value);
  }

  private StateMachine constructResult(Tasks tasks, ExtendedEventHandler listener) {
    Target target;
    try {
      target = pkg.getTarget(configuredTarget.getLabel().getName());
    } catch (NoSuchTargetException e) {
      // The package was fetched based on the label of the configured target. Since the configured
      // target exists, it must have existed in the package when it was created.
      throw new IllegalStateException("Target already verified for " + configuredTarget, e);
    }
    sink.acceptConfiguredTargetAndData(
        new ConfiguredTargetAndData(
            configuredTarget,
            target,
            configurationValue,
            transitionKey == null ? ImmutableList.of() : ImmutableList.of(transitionKey)),
        outputIndex);
    return DONE;
  }
}
