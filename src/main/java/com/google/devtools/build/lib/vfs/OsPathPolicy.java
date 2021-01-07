// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.devtools.build.lib.util.OS;

/**
 * An interface class representing the differences in path style between different OSs.
 *
 * <p>Eg. case sensitivity, '/' mounts vs. 'C:/', etc.
 */
public interface OsPathPolicy {
  int NORMALIZED = 0; // Path is normalized
  int NEEDS_NORMALIZE = 1; // Path requires normalization

  /** Returns required normalization level, passed to {@link #normalize}. */
  int needsToNormalize(String path);

  /**
   * Returns the required normalization level if an already normalized string is concatenated with
   * another normalized path fragment.
   *
   * <p>This method may be faster than {@link #needsToNormalize(String)}.
   */
  int needsToNormalizeSuffix(String normalizedSuffix);

  /**
   * Normalizes the passed string according to the passed normalization level.
   *
   * @param normalizationLevel The normalizationLevel from {@link #needsToNormalize}
   */
  String normalize(String path, int normalizationLevel);

  /**
   * Returns the length of the mount, eg. 1 for unix '/', 3 for Windows 'C:/'.
   *
   * <p>If the path is relative, 0 is returned
   */
  int getDriveStrLength(String path);

  /** Compares two path strings, using the given OS case sensitivity. */
  int compare(String s1, String s2);

  /** Compares two characters, using the given OS case sensitivity. */
  int compare(char c1, char c2);

  /** Tests two path strings for equality, using the given OS case sensitivity. */
  boolean equals(String s1, String s2);

  /** Computes the hash code for a path string. */
  int hash(String s);

  /**
   * Returns whether the passed string starts with the given prefix, given the OS case sensitivity.
   *
   * <p>This is a pure string operation and doesn't need to worry about matching path segments.
   */
  boolean startsWith(String path, String prefix);

  /**
   * Returns whether the passed string ends with the given prefix, given the OS case sensitivity.
   *
   * <p>This is a pure string operation and doesn't need to worry about matching path segments.
   */
  boolean endsWith(String path, String suffix);

  /** Returns whether the unnormalized character c is a separator. */
  boolean isSeparator(char c);

  /**
   * Returns an additional character besides '/' for which {@link #isSeparator} is true. 0 means
   * there is no such additional character.
   */
  char additionalSeparator();

  boolean isCaseSensitive();

  static OsPathPolicy getFilePathOs() {
    switch (OS.getCurrent()) {
      case LINUX:
      case FREEBSD:
      case OPENBSD:
      case UNKNOWN:
        return UnixOsPathPolicy.INSTANCE;
      case DARWIN:
        // NOTE: We *should* return a path policy that is case insensitive,
        // but we currently don't handle this
        return UnixOsPathPolicy.INSTANCE;
      case WINDOWS:
        return WindowsOsPathPolicy.INSTANCE;
      default:
        throw new AssertionError("Not covering all OSs");
    }
  }

  /** Utilities for implementations of {@link OsPathPolicy}. */
  class Utils {
    /**
     * Normalizes any '.' and '..' in-place in the segment array by shifting other segments to the
     * front. Returns the remaining number of items.
     */
    static int removeRelativePaths(String[] segments, int starti, boolean isAbsolute) {
      int segmentCount = 0;
      int shift = starti;
      int n = segments.length;
      for (int i = starti; i < n; ++i) {
        String segment = segments[i];
        switch (segment) {
          case ".":
            ++shift;
            break;
          case "..":
            if (segmentCount > 0 && !segments[segmentCount - 1].equals("..")) {
              // Remove the last segment, if there is one and it is not "..". This
              // means that the resulting path can still contain ".."
              // segments at the beginning.
              segmentCount--;
              shift += 2;
              break;
            } else if (isAbsolute) {
              // If this is absolute, then just pop it the ".." off and remain at root
              ++shift;
              break;
            }
            // Fall through
          default:
            ++segmentCount;
            if (shift > 0) {
              segments[i - shift] = segments[i];
            }
            break;
        }
      }
      return segmentCount;
    }
  }
}
