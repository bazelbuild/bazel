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

import com.google.auto.value.AutoBuilder;
import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.List;
import java.util.Map;

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
  public sealed interface MatchResult {
    /**
     * The configuration matches.
     *
     * <p>Preferably use the shared {@link MatchResult#MATCH} instance of this class.
     */
    @AutoCodec
    public record Match() implements MatchResult {}

    MatchResult MATCH = new Match();

    /**
     * The configuration does not match.
     *
     * @param diffs an optional list of diffs that describe the differences between the expected and
     *     actual configuration
     */
    @AutoCodec
    public record NoMatch(ImmutableList<Diff> diffs) implements MatchResult {
      @AutoCodec.Instantiator
      public NoMatch {}

      public NoMatch(Diff diff) {
        this(ImmutableList.of(diff));
      }

      /**
       * A human-readable description of the difference between the expected and actual
       * configuration.
       *
       * @param what the label of the constraint or setting that failed to match
       * @param got the actual value of the setting
       * @param want the expected value of the setting
       */
      @AutoCodec
      public record Diff(Label what, String got, String want) {
        public static Builder what(Label what) {
          return new AutoBuilder_ConfigMatchingProvider_MatchResult_NoMatch_Diff_Builder()
              .what(what);
        }

        /** A builder for {@link Diff}. */
        @AutoBuilder
        public abstract static class Builder {
          public abstract Builder what(Label what);

          public abstract Builder got(String got);

          public abstract Builder want(String want);

          public abstract Diff build();
        }
      }
    }

    /**
     * A result for the case in which an analysis error occurred that prevents the match from being
     * evaluated.
     */
    MatchResult ALREADY_REPORTED_NO_MATCH = new NoMatch(ImmutableList.of());

    /** Errors make the match question irresolvable. */
    @AutoCodec
    public record InError(ImmutableList<String> errors) implements MatchResult {}

    static MatchResult combine(MatchResult previous, MatchResult current) {
      return switch (previous) {
        // InError is the most severe state and always takes precedence.
        case InError(ImmutableList<String> previousErrors) ->
            switch (current) {
              case InError(ImmutableList<String> currentErrors) ->
                  new InError(
                      ImmutableList.<String>builder()
                          .addAll(previousErrors)
                          .addAll(currentErrors)
                          .build());
              default -> previous;
            };
        case NoMatch(ImmutableList<NoMatch.Diff> previousDiffs) ->
            switch (current) {
              case InError(ImmutableList<String> ignored) -> current;
              case NoMatch(ImmutableList<NoMatch.Diff> currentDiffs) ->
                  new NoMatch(
                      ImmutableList.<NoMatch.Diff>builder()
                          .addAll(previousDiffs)
                          .addAll(currentDiffs)
                          .build());
              case Match() -> previous;
            };
        case Match ignored -> current;
      };
    }
  }

  /** Result of accumulating match results: contains any errors or non-matching labels. */
  public record AccumulateResults(
      ImmutableList<Label> nonMatching, ImmutableMultimap<Label, String> errors) {
    public boolean success() {
      return nonMatching.isEmpty() && errors.isEmpty();
    }
  }

  /**
   * Combine the results from the given {@link ConfigMatchingProvider} instances, returning any
   * errors and non-matching providers.
   */
  public static AccumulateResults accumulateMatchResults(List<ConfigMatchingProvider> providers) {
    ImmutableList.Builder<Label> nonMatching = ImmutableList.builder();
    ImmutableMultimap.Builder<Label, String> errors = ImmutableMultimap.builder();
    for (ConfigMatchingProvider configProvider : providers) {
      MatchResult matchResult = configProvider.result();
      if (matchResult instanceof MatchResult.InError(ImmutableList<String> messages)) {
        errors.putAll(configProvider.label(), messages);
      } else if (matchResult instanceof MatchResult.NoMatch) {
        nonMatching.add(configProvider.label());
      }
    }

    return new AccumulateResults(nonMatching.build(), errors.build());
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
