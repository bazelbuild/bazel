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

package com.google.devtools.build.lib.skyframe.trimming;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A simple key suitable for putting into a TrimmedConfigurationCache for testing it. */
@AutoValue
abstract class TestKey {
  abstract String descriptor();

  abstract ImmutableMap<String, String> configuration();

  static TestKey create(String descriptor, ImmutableMap<String, String> configuration) {
    return new AutoValue_TestKey(descriptor, configuration);
  }

  // Test keys look like <config: value, config: value> descriptor
  private static final Pattern TEST_KEY_SHAPE =
      Pattern.compile("<(?<config>[^>]*)>(?<descriptor>.+)");

  static TestKey parse(String input) {
    Matcher matcher = TEST_KEY_SHAPE.matcher(input.trim());
    Preconditions.checkArgument(matcher.matches());
    return create(
        matcher.group("descriptor").trim(), parseConfiguration(matcher.group("config").trim()));
  }

  static ImmutableMap<String, String> parseConfiguration(String input) {
    if (Strings.isNullOrEmpty(input)) {
      return ImmutableMap.of();
    }
    return ImmutableMap.copyOf(
        Splitter.on(',')
            .trimResults()
            .withKeyValueSeparator(Splitter.on(':').trimResults())
            .split(input));
  }

  static ConfigurationComparer.Result compareConfigurations(
      ImmutableMap<String, String> left,
      ImmutableMap<String, String> right) {
    Set<String> sharedKeys = Sets.intersection(left.keySet(), right.keySet());
    for (String key : sharedKeys) {
      if (!left.get(key).equals(right.get(key))) {
        return ConfigurationComparer.Result.DIFFERENT;
      }
    }
    boolean hasLeftOnlyKeys = !Sets.difference(left.keySet(), right.keySet()).isEmpty();
    boolean hasRightOnlyKeys = !Sets.difference(right.keySet(), left.keySet()).isEmpty();
    if (hasLeftOnlyKeys) {
      if (hasRightOnlyKeys) {
        return ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL;
      } else {
        return ConfigurationComparer.Result.SUPERSET;
      }
    } else {
      if (hasRightOnlyKeys) {
        return ConfigurationComparer.Result.SUBSET;
      } else {
        return ConfigurationComparer.Result.EQUAL;
      }
    }
  }

  /** Produces a cache suitable for storing TestKeys. */
  static TrimmedConfigurationCache<TestKey, String, ImmutableMap<String, String>> newCache() {
    return new TrimmedConfigurationCache<>(
        TestKey::descriptor, TestKey::configuration, TestKey::compareConfigurations);
  }
}
