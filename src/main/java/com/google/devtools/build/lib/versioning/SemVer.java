// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.versioning;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.auto.value.AutoValue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents semantic versions (semver) and allows parsing them.
 *
 * <p>This implementation is supposed to be program-agnostic and thus everything it supports should
 * be applicable to any program following semver.
 */
// TODO(jmmv): Would be nice to take com.google.devtools.build.lib.rules.apple.DottedVersion and
// build the oddities of that scheme on top of this, given that the differences are minimal.
@AutoValue
public abstract class SemVer implements Comparable<SemVer> {

  /** The major version stored in this semver instance. */
  abstract int major();

  /** The minor version stored in this semver instance. */
  abstract int minor();

  /** The patch version stored in this semver instance. */
  abstract int patch();

  /** Constructs a new semver from the given components with minor and patch set to 0. */
  public static SemVer from(int major) {
    return from(major, 0, 0);
  }

  /** Constructs a new semver from the given components with patch set to 0. */
  public static SemVer from(int major, int minor) {
    return from(major, minor, 0);
  }

  /** Constructs a new semver from the given components. */
  public static SemVer from(int major, int minor, int patch) {
    checkArgument(major >= 0);
    checkArgument(minor >= 0);
    checkArgument(patch >= 0);
    return new AutoValue_SemVer(major, minor, patch);
  }

  /** Parses a semver from a string. */
  public static SemVer parse(String text) throws ParseException {
    Pattern pattern = Pattern.compile("([0-9]+)(.([0-9]+)(.([0-9]+))?)?");
    Matcher matcher = pattern.matcher(text);
    if (!matcher.matches()) {
      throw new ParseException("Invalid semver " + text);
    }
    try {
      int major = Integer.parseInt(matcher.group(1));
      String maybeMinor = matcher.group(3);
      String maybePatch = matcher.group(5);
      if (maybePatch != null) {
        return from(major, Integer.parseInt(maybeMinor), Integer.parseInt(maybePatch));
      } else if (matcher.group(3) != null) {
        return from(major, Integer.parseInt(maybeMinor));
      } else {
        return from(major);
      }
    } catch (NumberFormatException e) {
      throw new ParseException("Invalid number in semver component", e);
    }
  }

  @Override
  public int compareTo(SemVer o) {
    int majorRelation = Integer.compare(major(), o.major());
    int minorRelation = Integer.compare(minor(), o.minor());
    int patchRelation = Integer.compare(patch(), o.patch());
    return majorRelation != 0
        ? majorRelation
        : (minorRelation != 0 ? minorRelation : patchRelation);
  }

  @Override
  public final String toString() {
    return String.format("%d.%d.%d", major(), minor(), patch());
  }
}
