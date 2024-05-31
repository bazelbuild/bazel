// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A "configuration target" that asserts whether or not it matches the configuration it's bound to.
 *
 * <p>This can be used, e.g., to declare a BUILD target that defines the conditions which trigger a
 * configurable attribute branch. In general, this can be used to trigger for any user-configurable
 * build behavior.
 */
@Immutable
@AutoValue
public abstract class ConfigMatchingProvider implements TransitiveInfoProvider {
  /**
   * Potential values for result field.
   *
   * <p>Note that while it is possible to be more aggressive in interpreting and merging
   * MatchResult, currently taking a more cautious approach and focusing on propagating errors.
   *
   * <p>e.g. If merging where one is InError and the other is No, then currently will propagate the
   * errors, versus a more aggressive future approach could just propagate No.)
   */
  // TODO(twigg): This is more cleanly implemented when Java record is available.
  public static interface MatchResult {
    // Only InError should return non-null.
    public abstract @Nullable String getError();

    public static final MatchResult MATCH = HasResult.MATCH;
    public static final MatchResult NOMATCH = HasResult.NOMATCH;

    /** Some specified error makes the match question irresolvable. */
    @AutoValue
    public abstract class InError implements MatchResult {
      public static InError create(String error) {
        return new AutoValue_ConfigMatchingProvider_MatchResult_InError(error);
      }
    }

    public static MatchResult create(boolean matches) {
      return matches ? MATCH : NOMATCH;
    }

    /** If previously InError or No, keep previous else convert newData. */
    public static MatchResult merge(MatchResult previousMatch, boolean andWith) {
      if (!previousMatch.equals(MATCH)) {
        return previousMatch;
      }
      return andWith ? MATCH : NOMATCH;
    }
  }

  /** Result of accumulating match results: contains any errors or non-matching labels. */
  public record AccumulateResults(
      ImmutableList<Label> nonMatching, ImmutableMap<Label, String> errors) {
    public boolean success() {
      return nonMatching.isEmpty() && errors.isEmpty();
    }
  }

  /**
   * Combine the results from the given {@link ConfigMatchingProvider} instances, returning any
   * errors and non-matching providers.
   */
  public static AccumulateResults accumulateMatchResults(List<ConfigMatchingProvider> providers) {
    ImmutableList.Builder<Label> nonMatching = new ImmutableList.Builder<>();
    ImmutableMap.Builder<Label, String> errors = new ImmutableMap.Builder<>();
    for (ConfigMatchingProvider configProvider : providers) {
      ConfigMatchingProvider.MatchResult matchResult = configProvider.result();
      if (matchResult.getError() != null) {
        String message = matchResult.getError();
        errors.put(configProvider.label(), message);
      } else if (matchResult.equals(ConfigMatchingProvider.MatchResult.NOMATCH)) {
        nonMatching.add(configProvider.label());
      }
    }

    return new AccumulateResults(nonMatching.build(), errors.buildKeepingLast());
  }

  /**
   * The result was resolved.
   *
   * <p>Using an enum to get convenient toString, equals, and hashCode implementations. Interfaces
   * can't have private members so this is defined here privately and then exported into the
   * MatchResult interface to allow for ergonomic MatchResult.MATCH checks.
   */
  private static enum HasResult implements MatchResult {
    MATCH,
    NOMATCH;

    @Override
    public @Nullable String getError() {
      return null;
    }
  }

  /**
   * Create a ConfigMatchingProvider.
   *
   * @param label the build label corresponding to this matcher
   * @param settingsMap the condition settings that trigger this matcher
   * @param flagSettingsMap the label-keyed settings that trigger this matcher
   * @param result whether the current associated configuration matches, doesn't match, or is
   *     irresolvable due to specified issue
   */
  public static ConfigMatchingProvider create(
      Label label,
      ImmutableMultimap<String, String> settingsMap,
      ImmutableMap<Label, String> flagSettingsMap,
      ImmutableSet<Label> constraintValueSettings,
      MatchResult result) {
    return new AutoValue_ConfigMatchingProvider(
        label, settingsMap, flagSettingsMap, constraintValueSettings, result);
  }

  /** The target's label. */
  public abstract Label label();

  public abstract ImmutableMultimap<String, String> settingsMap();

  public abstract ImmutableMap<Label, String> flagSettingsMap();

  public abstract ImmutableSet<Label> constraintValuesSetting();

  /**
   * Whether or not the configuration criteria defined by this target match its actual
   * configuration.
   */
  public abstract MatchResult result();

  /**
   * Returns true if this matcher's conditions are a proper superset of another matcher's
   * conditions, i.e. if this matcher is a specialization of the other one.
   */
  public boolean refines(ConfigMatchingProvider other) {
    ImmutableSet<Map.Entry<String, String>> settings = ImmutableSet.copyOf(settingsMap().entries());
    ImmutableSet<Map.Entry<String, String>> otherSettings =
        ImmutableSet.copyOf(other.settingsMap().entries());
    ImmutableSet<Map.Entry<Label, String>> flagSettings = flagSettingsMap().entrySet();
    ImmutableSet<Map.Entry<Label, String>> otherFlagSettings = other.flagSettingsMap().entrySet();

    ImmutableSet<Label> constraintValueSettings = constraintValuesSetting();
    ImmutableSet<Label> otherConstraintValueSettings = other.constraintValuesSetting();

    if (!settings.containsAll(otherSettings)
        || !flagSettings.containsAll(otherFlagSettings)
        || !constraintValueSettings.containsAll(otherConstraintValueSettings)) {
      return false; // Not a superset.
    }

    return settings.size() > otherSettings.size()
        || flagSettings.size() > otherFlagSettings.size()
        || constraintValueSettings.size() > otherConstraintValueSettings.size();
  }

  /** Format this provider as its label. */
  @Override
  public final String toString() {
    return label().toString();
  }
}
