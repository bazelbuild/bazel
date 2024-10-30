// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider.AccumulateResults;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** Helper class for {@link ConfigMatchingProvider} data. */
public final class ConfigMatchingUtil {
  private ConfigMatchingUtil() {}

  /**
   * Validates that the given {@link ConfigMatchingProvider} instances are all successful.
   *
   * @return whether all instances are successful
   * @param label the source of the instances, for error reporting
   * @param configSettings the instances to check
   * @param errorHandler if non-null, an error message will be generated for each non-successful
   *     instance
   * @throws InvalidConfigurationException thrown if any instances are in an error state
   */
  public static boolean validate(
      Label label,
      ImmutableList<ConfigMatchingProvider> configSettings,
      @Nullable Consumer<String> errorHandler)
      throws InvalidConfigurationException {
    // Make sure the target setting matches but watch out for resolution errors.
    AccumulateResults accumulateResults =
        ConfigMatchingProvider.accumulateMatchResults(configSettings);
    if (!accumulateResults.errors().isEmpty()) {
      // TODO(blaze-configurability-team): This should only be due to feature flag trimming. So,
      // would be better to just ensure toolchain resolution isn't transitively dependent on
      // feature flags at all.
      String message =
          accumulateResults.errors().entrySet().stream()
              .map(
                  entry ->
                      String.format(
                          "For config_setting %s, %s", entry.getKey().getName(), entry.getValue()))
              .collect(joining("; "));
      throw new InvalidConfigurationException(
          "Unrecoverable errors resolving config_setting associated with "
              + label
              + ": "
              + message);
    }
    if (accumulateResults.success()) {
      return true;
    } else if (!accumulateResults.nonMatching().isEmpty() && errorHandler != null) {
      String nonMatchingList =
          accumulateResults.nonMatching().stream()
              .distinct()
              .map(Label::getName)
              .collect(joining(", "));
      String message = String.format("mismatching config settings: %s", nonMatchingList);
      errorHandler.accept(message);
    }
    return false;
  }
}
