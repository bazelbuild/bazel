// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.toolchains;

import static java.util.stream.Collectors.joining;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.AbstractLabelCycleReporter;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.toolchains.SingleToolchainResolutionValue.SingleToolchainResolutionKey;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Optional;

/**
 * {@link CyclesReporter.SingleCycleReporter} implementation that can handle cycles involving
 * registered toolchains.
 */
public class RegisteredToolchainsCycleReporter implements CyclesReporter.SingleCycleReporter {

  private static final Predicate<SkyKey> IS_REGISTERED_TOOLCHAINS_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.REGISTERED_TOOLCHAINS);

  private static final Predicate<SkyKey> IS_CONFIGURED_TARGET_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET);

  private static final Predicate<SkyKey> IS_SINGLE_TOOLCHAIN_RESOLUTION_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.SINGLE_TOOLCHAIN_RESOLUTION);

  private static final Predicate<SkyKey> IS_TOOLCHAIN_RESOLUTION_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.TOOLCHAIN_RESOLUTION);

  private static final Predicate<SkyKey> IS_TOOLCHAIN_RELATED =
      Predicates.or(
          IS_REGISTERED_TOOLCHAINS_SKY_KEY,
          IS_SINGLE_TOOLCHAIN_RESOLUTION_SKY_KEY,
          IS_TOOLCHAIN_RESOLUTION_SKY_KEY);

  @Override
  public boolean maybeReportCycle(
      SkyKey topLevelKey,
      CycleInfo cycleInfo,
      boolean alreadyReported,
      ExtendedEventHandler eventHandler) {
    ImmutableList<SkyKey> cycle = cycleInfo.getCycle();
    if (alreadyReported) {
      return true;
    } else if (!Iterables.any(cycle, IS_TOOLCHAIN_RELATED)) {
      return false;
    }

    // Find the ConfiguredTargetKey, this should tell the problem.
    Optional<ConfiguredTargetKey> configuredTargetKey = findRootConfiguredTarget(cycle);
    if (!configuredTargetKey.isPresent()) {
      return false;
    }

    Function<Object, String> printer =
        input -> {
          if (input instanceof ConfiguredTargetKey ctk) {
            Label label = ctk.getLabel();
            return label.toString();
          }
          if (input instanceof RegisteredToolchainsValue.Key) {
            return "RegisteredToolchains";
          }
          if (input instanceof SingleToolchainResolutionKey) {
            Label toolchainType =
                ((SingleToolchainResolutionKey) input).toolchainType().toolchainType();
            return String.format("toolchain type %s", toolchainType);
          }
          if (input instanceof ToolchainContextKey toolchainContextKey) {
            String toolchainTypes =
                toolchainContextKey.toolchainTypes().stream()
                    .map(ToolchainTypeRequirement::toolchainType)
                    .map(Label::toString)
                    .collect(joining(", "));
            return String.format("toolchain types %s", toolchainTypes);
          }

          throw new UnsupportedOperationException(input.toString());
        };

    StringBuilder cycleMessage =
        new StringBuilder()
            .append("Misconfigured toolchains: ")
            .append(printer.apply(configuredTargetKey.get()))
            .append(" is declared as a toolchain but has inappropriate dependencies.")
            .append(" Declared toolchains should be created with the 'toolchain' rule")
            .append(" and should not have dependencies that themselves require toolchains.");

    AbstractLabelCycleReporter.printCycle(cycleInfo.getCycle(), cycleMessage, printer);
    eventHandler.handle(Event.error(null, cycleMessage.toString()));
    return true;
  }

  /**
   * Returns the first {@link SkyKey} that is an instance of {@link ConfiguredTargetKey} and follows
   * {@link RegisteredToolchainsValue.Key}. This will loop over the cycle in case the {@link
   * RegisteredToolchainsValue} is not first in the list.
   */
  private Optional<ConfiguredTargetKey> findRootConfiguredTarget(ImmutableList<SkyKey> cycle) {
    // Loop over the cycle, possibly twice, first looking for RegisteredToolchainsValue,
    // then finding the first ConfiguredTargetKey.
    boolean rtvFound = false;
    for (int i = 0; i < cycle.size() * 2; i++) {
      SkyKey skyKey = cycle.get(i % cycle.size());
      if (!rtvFound && IS_REGISTERED_TOOLCHAINS_SKY_KEY.apply(skyKey)) {
        rtvFound = true;
      }
      if (rtvFound && IS_CONFIGURED_TARGET_SKY_KEY.apply(skyKey)) {
        return Optional.of((ConfiguredTargetKey) skyKey);
      }
    }

    return Optional.empty();
  }
}
