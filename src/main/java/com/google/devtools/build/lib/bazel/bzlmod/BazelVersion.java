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
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import java.util.Comparator;
import java.util.Objects;
import java.util.Optional;
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

  /* Valid bazel compatibility argument must 1) start with (<,<=,>,>=,-);
     2) then contain a version number in form of X.X.X where X has one or two digits
  */
  private static final Pattern VALID_BAZEL_COMPATIBILITY_VERSION =
      Pattern.compile("(>|<|-|<=|>=)(\\d+\\.){2}\\d+");

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

  private static BazelVersion getInstance() {
    return parse(BlazeVersionInfo.instance().getVersion());
  }

  @Nullable
  private static BazelVersion parse(String version) {
    // For a prerelease version of Bazel, pretend that all constraints are satisfied (see
    // satisfiesCompatibility below).
    if (version.isEmpty()) {
      return null;
    }
    Matcher matcher = PATTERN.matcher(version);
    Preconditions.checkArgument(
        matcher.matches(), "bad version (does not match regex): %s", version);

    String release = matcher.group("release");
    @Nullable String suffix = matcher.group("suffix");

    ImmutableList<Integer> releaseSplit =
        Splitter.on('.').splitToStream(release).map(Integer::valueOf).collect(toImmutableList());
    return new AutoValue_BazelVersion(releaseSplit, nullToEmpty(suffix), version);
  }

  /** Returns the current Bazel version as a string, or "<dev>" if unknown. */
  public static String getCurrentVersionString() {
    var currentVersion = getInstance();
    return currentVersion == null ? "<dev>" : currentVersion.getOriginal();
  }

  /**
   * Check if the current Bazel version satisfies the given compatibility version constraints and
   * returns an error message if not.
   */
  public static Optional<String> checkCompatibility(
      ImmutableList<String> bazelCompatibility, ModuleKey moduleKey) {
    var currentVersion = getInstance();
    if (currentVersion == null) {
      return Optional.empty();
    }

    for (String compatVersion : bazelCompatibility) {
      if (!VALID_BAZEL_COMPATIBILITY_VERSION.matcher(compatVersion).matches()) {
        return Optional.of(
            ("invalid version argument '%s': valid argument must 1) start with (<,<=,>,>=,-); 2)"
                    + " contain a version number in form of X.X.X where X is a number")
                .formatted(compatVersion));
      }
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
              currentVersion.getRelease(),
              compatSplit,
              lexicographical(Comparator.<Integer>naturalOrder()));
      if (result == 0 && currentVersion.isPrerelease()) {
        result = -1;
      }

      if (!((result == 0 && sign.contains("="))
          || (result > 0 && (sign.contains(">") || sign.contains("-")))
          || (result < 0 && (sign.contains("<") || sign.contains("-"))))) {
        return Optional.of(
            "Bazel version %s is not compatible with module \"%s\" (bazel_compatibility: %s)"
                .formatted(currentVersion.getOriginal(), moduleKey, bazelCompatibility));
      }
    }
    return Optional.empty();
  }
}
