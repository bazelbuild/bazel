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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;

@VisibleForTesting
class UnixOsPathPolicy implements OsPathPolicy {

  static final UnixOsPathPolicy INSTANCE = new UnixOsPathPolicy();
  private static final Splitter PATH_SPLITTER = Splitter.onPattern("/+").omitEmptyStrings();

  @Override
  public int needsToNormalize(String path) {
    int n = path.length();
    int dotCount = 0;
    char prevChar = 0;
    for (int i = 0; i < n; i++) {
      char c = path.charAt(i);
      if (c == '\\') {
        return NEEDS_NORMALIZE;
      }
      if (c == '/') {
        if (prevChar == '/') {
          return NEEDS_NORMALIZE;
        }
        if (dotCount == 1 || dotCount == 2) {
          return NEEDS_NORMALIZE;
        }
      }
      dotCount = c == '.' ? dotCount + 1 : 0;
      prevChar = c;
    }
    if (prevChar == '/' || dotCount == 1 || dotCount == 2) {
      return NEEDS_NORMALIZE;
    }
    return NORMALIZED;
  }

  @Override
  public int needsToNormalizeSuffix(String normalizedSuffix) {
    // We know that the string is normalized
    // In this case only suffixes starting with ".." may cause
    // normalization once concatenated with other strings
    return normalizedSuffix.startsWith("..") ? NEEDS_NORMALIZE : NORMALIZED;
  }

  @Override
  public String normalize(String path, int normalizationLevel) {
    if (normalizationLevel == NORMALIZED) {
      return path;
    }
    if (path.isEmpty()) {
      return path;
    }
    boolean isAbsolute = path.charAt(0) == '/';
    StringBuilder sb = new StringBuilder(path.length());
    if (isAbsolute) {
      sb.append('/');
    }
    String[] segments = Iterables.toArray(PATH_SPLITTER.splitToList(path), String.class);
    int segmentCount = Utils.removeRelativePaths(segments, 0, isAbsolute);
    for (int i = 0; i < segmentCount; ++i) {
      sb.append(segments[i]);
      sb.append('/');
    }
    if (segmentCount > 0) {
      sb.deleteCharAt(sb.length() - 1);
    }
    return sb.toString();
  }

  @Override
  public int getDriveStrLength(String path) {
    if (path.length() == 0) {
      return 0;
    }
    return (path.charAt(0) == '/') ? 1 : 0;
  }

  @Override
  public int compare(String s1, String s2) {
    return s1.compareTo(s2);
  }

  @Override
  public int compare(char c1, char c2) {
    return Character.compare(c1, c2);
  }

  @Override
  public boolean equals(String s1, String s2) {
    return s1.equals(s2);
  }

  @Override
  public int hash(String s) {
    return s.hashCode();
  }

  @Override
  public boolean startsWith(String path, String prefix) {
    return path.startsWith(prefix);
  }

  @Override
  public boolean endsWith(String path, String suffix) {
    return path.endsWith(suffix);
  }

  @Override
  public boolean isSeparator(char c) {
    return c == '/';
  }

  @Override
  public char additionalSeparator() {
    return 0;
  }

  @Override
  public boolean isCaseSensitive() {
    return true;
  }
}
