// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skymetal;

import static java.util.Objects.requireNonNull;

import com.google.common.flogger.GoogleLogger;
import java.util.function.Function;

/**
 * Feature flag descriptor and evaluation for incremental Skymetal native migrations.
 *
 * <p>Each migrated module defines a single {@code public static final SkymetalFeature SKYMETAL =
 * SkymetalFeature.of("<feature>", Stage.<STAGE>, ...)} (declaring any prerequisite {@link
 * SkymetalFeature}s it depends on) and checks {@code SKYMETAL.isEnabled()} at branch sites. Because
 * {@code SkymetalFeature} is a {@code record}, HotSpot's JIT treats {@code SKYMETAL.isEnabled()} on
 * a {@code static final} instance as a compile-time constant and dead-code-eliminates inactive
 * paths.
 *
 * <p>By default (when {@code -Dskymetal} is unset), the active tier is {@code "stable"} and all
 * {@link Stage#STABLE} features are enabled. Individual features can be toggled via {@code
 * -Dskymetal.<feature>=true|false}. Disabling a prerequisite feature automatically disables
 * dependent features (logging a warning if a dependent feature was explicitly requested with {@code
 * -Dskymetal.<dependent>=true}). Once a feature graduates, it is always enabled and its Java
 * fallback path and {@code SKYMETAL} constant are deleted.
 *
 * @param name the feature name used for {@code -Dskymetal.<name>} overrides
 * @param stage the rollout maturity tier of the feature
 * @param isEnabled whether the feature is enabled for the current JVM process
 */
public record SkymetalFeature(String name, Stage stage, boolean isEnabled) {
  /** Rollout maturity tier for a Skymetal feature. */
  public enum Stage {
    /** Enabled by default (when {@code -Dskymetal} is unset or {@code "stable"}). */
    STABLE,
    /** Enabled when {@code -Dskymetal=preview} or {@code -Dskymetal=experimental}. */
    PREVIEW,
    /** Enabled when {@code -Dskymetal=experimental}. */
    EXPERIMENTAL;

    boolean isEnabledAt(Stage activeStage) {
      return switch (activeStage) {
        case STABLE -> this == STABLE;
        case PREVIEW -> this == STABLE || this == PREVIEW;
        case EXPERIMENTAL -> true;
      };
    }

    private static Stage parse(String value) {
      return switch (value) {
        case "stable" -> STABLE;
        case "preview" -> PREVIEW;
        case "experimental" -> EXPERIMENTAL;
        default ->
            throw new IllegalArgumentException(
                "Unknown -Dskymetal=" + value + " (expected: stable, preview, or experimental)");
      };
    }
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public SkymetalFeature {
    requireNonNull(name, "name");
    requireNonNull(stage, "stage");
    if (name.isEmpty()) {
      throw new IllegalArgumentException("Feature name must not be empty");
    }
  }

  /**
   * Defines and evaluates a {@link SkymetalFeature} with optional prerequisite {@code
   * dependencies}.
   */
  public static SkymetalFeature of(String name, Stage stage, SkymetalFeature... dependencies) {
    return of(name, stage, System::getProperty, dependencies);
  }

  static SkymetalFeature of(
      String name,
      Stage stage,
      Function<String, String> getProperty,
      SkymetalFeature... dependencies) {
    boolean enabled = evaluateEnabled(name, stage, getProperty, dependencies);
    return new SkymetalFeature(name, stage, enabled);
  }

  private static boolean evaluateEnabled(
      String name,
      Stage stage,
      Function<String, String> getProperty,
      SkymetalFeature... dependencies) {
    requireNonNull(name, "name");
    requireNonNull(stage, "stage");
    requireNonNull(getProperty, "getProperty");
    requireNonNull(dependencies, "dependencies");
    if (name.isEmpty()) {
      throw new IllegalArgumentException("Feature name must not be empty");
    }

    // 1. Validate the active tier upfront, then check explicit per-feature override
    // (-Dskymetal.<feature>=true|false) or the active tier.
    String activeStageProp = getProperty.apply("skymetal");
    Stage activeStage = activeStageProp == null ? Stage.STABLE : Stage.parse(activeStageProp);

    String explicitProp = getProperty.apply("skymetal." + name);
    boolean wantsEnabled =
        explicitProp != null
            ? parseStrictBoolean("skymetal." + name, explicitProp)
            : stage.isEnabledAt(activeStage);
    if (!wantsEnabled) {
      return false;
    }

    // 2. Verify all prerequisite features are enabled when this feature wants to be enabled.
    for (SkymetalFeature dependency : dependencies) {
      requireNonNull(dependency, "dependency");
      if (!dependency.isEnabled()) {
        if (explicitProp != null) {
          logger.atWarning().log(
              "Cannot enable Skymetal feature '%s' (-Dskymetal.%s=true) because prerequisite '%s'"
                  + " is disabled",
              name, name, dependency.name());
        }
        return false;
      }
    }
    return true;
  }

  private static boolean parseStrictBoolean(String propertyName, String value) {
    return switch (value) {
      case "true" -> true;
      case "false" -> false;
      default ->
          throw new IllegalArgumentException(
              "Invalid -D" + propertyName + "=" + value + " (expected: true or false)");
    };
  }
}
