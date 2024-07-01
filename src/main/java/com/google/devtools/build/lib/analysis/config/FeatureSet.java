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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Maps;
import com.google.common.collect.Streams;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Represents a set of "on" features and a set of "off" features. The two sets are guaranteed not to
 * intersect.
 */
@AutoValue
public abstract class FeatureSet {
  public static final FeatureSet EMPTY = of(ImmutableSet.of(), ImmutableSet.of());

  public abstract ImmutableSet<String> on();

  public abstract ImmutableSet<String> off();

  private static FeatureSet of(Set<String> on, Set<String> off) {
    return new AutoValue_FeatureSet(ImmutableSortedSet.copyOf(on), ImmutableSortedSet.copyOf(off));
  }

  /** Parses a {@link FeatureSet} instance from a list of strings. */
  public static FeatureSet parse(Iterable<String> features) {
    for (String feature : features) {
      Preconditions.checkArgument(
          !feature.contains(","),
          String.format("Feature %s contains a comma `,`. If provided via the command-line, use multiple --features instead.", feature));
    }
    Map<String, Boolean> featureToState = new HashMap<>();
    for (String feature : features) {
      if (feature.startsWith("-")) {
        featureToState.put(feature.substring(1), false);
      } else if (feature.equals("no_layering_check")) {
        // TODO(bazel-team): Remove once we do not have BUILD files left that contain
        // 'no_layering_check'.
        featureToState.put("layering_check", false);
      } else {
        // -X always trumps X.
        featureToState.putIfAbsent(feature, true);
      }
    }
    return fromMap(featureToState);
  }

  private static FeatureSet fromMap(Map<String, Boolean> featureToState) {
    return of(
        Maps.filterValues(featureToState, Boolean.TRUE::equals).keySet(),
        Maps.filterValues(featureToState, Boolean.FALSE::equals).keySet());
  }

  private static void mergeSetIntoMap(
      Set<String> features, boolean state, Map<String, Boolean> featureToState) {
    for (String feature : features) {
      featureToState.put(feature, state);
    }
  }

  /**
   * Merges two {@link FeatureSet}s into one, with {@code coarse} being the coarser-grained set
   * (e.g. the package default feature set), and {@code fine} being the finer-grained set (e.g. the
   * rule-level feature set). Note that this operation is not commutative.
   */
  public static FeatureSet merge(FeatureSet coarse, FeatureSet fine) {
    Map<String, Boolean> featureToState = new HashMap<>();
    mergeSetIntoMap(coarse.on(), true, featureToState);
    mergeSetIntoMap(coarse.off(), false, featureToState);
    mergeSetIntoMap(fine.on(), true, featureToState);
    mergeSetIntoMap(fine.off(), false, featureToState);
    return fromMap(featureToState);
  }

  /**
   * Merges a {@link FeatureSet} with the global feature set. This differs from {@link #merge} in
   * that the globally disabled features are <strong>always</strong> disabled.
   */
  public static FeatureSet mergeWithGlobalFeatures(FeatureSet base, FeatureSet global) {
    Map<String, Boolean> featureToState = new HashMap<>();
    mergeSetIntoMap(global.on(), true, featureToState);
    mergeSetIntoMap(base.on(), true, featureToState);
    mergeSetIntoMap(base.off(), false, featureToState);
    mergeSetIntoMap(global.off(), false, featureToState);
    return fromMap(featureToState);
  }

  public final ImmutableList<String> toStringList() {
    return Streams.concat(on().stream(), off().stream().map(s -> "-" + s))
        .collect(toImmutableList());
  }
}
