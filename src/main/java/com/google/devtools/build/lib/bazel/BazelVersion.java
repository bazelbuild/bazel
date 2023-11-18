// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel;

import static com.google.common.base.Strings.nullToEmpty;
import static com.google.common.collect.Comparators.lexicographical;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import java.util.Comparator;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Represents a bazel version. The version format supported is {RELEASE[SUFFIX]}, where:
 *
 * <ul>
 *   <li>{RELEASE} is a sequence of decimal numbers separated by dots;
 *   <li>{SUFFIX} could be: {@code -pre.*}, or any other string (which compares equal to SUFFIX
 *       being absent)
 * </ul>
 */
@AutoValue
public abstract class BazelVersion {

  private static final Pattern PATTERN =
      Pattern.compile("(?<release>(?:\\d+\\.)*\\d+)(?<suffix>(.*))?");

  /** Returns the "release" part of the version string as a list of integers. */
  abstract ImmutableList<Integer> getRelease();

  /** Returns the "suffix" part of the version that starts after the integers */
  abstract String getSuffix();

  /** Returns the original version string. */
  public abstract String getOriginal();

  /** Whether this is a prerelease */
  boolean isPrerelease() {
    return getSuffix().startsWith("-pre");
  }

  /** Parses a version string into a {@link BazelVersion} object. */
  public static BazelVersion parse(String version) {
    Matcher matcher = PATTERN.matcher(version);
    Preconditions.checkArgument(
        matcher.matches(), "bad version (does not match regex): %s", version);

    String release = matcher.group("release");
    @Nullable String suffix = matcher.group("suffix");

    ImmutableList<Integer> releaseSplit =
        Splitter.on('.').splitToStream(release).map(Integer::valueOf).collect(toImmutableList());
    return new AutoValue_BazelVersion(releaseSplit, nullToEmpty(suffix), version);
  }

  /** Check if class version satisfies compatibility version */
  public boolean satisfiesCompatibility(String compatVersion) {
    int cutIndex = compatVersion.contains("=") ? 2 : 1;
    String sign = compatVersion.substring(0, cutIndex);
    compatVersion = compatVersion.substring(cutIndex);

    ImmutableList<Integer> compatSplit =
        Splitter.on('.')
            .splitToStream(compatVersion)
            .map(Integer::valueOf)
            .collect(toImmutableList());

    int result =
        Objects.compare(
            getRelease(), compatSplit, lexicographical(Comparator.<Integer>naturalOrder()));
    if (result == 0 && isPrerelease()) {
      result = -1;
    }

    return (result == 0 && sign.contains("="))
        || (result > 0 && (sign.contains(">") || sign.contains("-")))
        || (result < 0 && (sign.contains("<") || sign.contains("-")));
  }
}
