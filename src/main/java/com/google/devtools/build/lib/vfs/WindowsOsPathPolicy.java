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
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.WindowsPathOperations;
import java.io.IOException;

@VisibleForTesting
class WindowsOsPathPolicy implements OsPathPolicy {

  static final WindowsOsPathPolicy INSTANCE =
      new WindowsOsPathPolicy(new DefaultShortPathResolver());

  static final WindowsOsPathPolicy CROSS_PLATFORM_INSTANCE =
      new WindowsOsPathPolicy(new CrossPlatformShortPathResolver());

  static final int NEEDS_SHORT_PATH_NORMALIZATION = NEEDS_NORMALIZE + 1;

  private static final Splitter WINDOWS_PATH_SPLITTER =
      Splitter.onPattern("[\\\\/]+").omitEmptyStrings();

  private final ShortPathResolver shortPathResolver;

  interface ShortPathResolver {
    String resolveShortPath(String path);
  }

  static class DefaultShortPathResolver implements ShortPathResolver {
    @Override
    public String resolveShortPath(String path) {
      if (!OS.getCurrent().equals(OS.WINDOWS)) {
        // Short path resolution only makes sense on a Windows host.
        return path;
      }
      try {
        return WindowsPathOperations.getLongPath(path);
      } catch (IOException e) {
        return path;
      }
    }
  }

  static class CrossPlatformShortPathResolver implements ShortPathResolver {
    @Override
    public String resolveShortPath(String path) {
      // Short paths can only be resolved on a Windows host.
      // Skipping short path resolution when running on a non-Windows host can
      // result in paths considered different that are actually the same.
      // TODO: Consider failing when a short path is detected on a non-Windows
      //  host. Since short path segments can arise from most operations on
      //  PathFragment, this would however require exception handling in many
      //  places.
      return path;
    }
  }

  @VisibleForTesting
  WindowsOsPathPolicy(ShortPathResolver shortPathResolver) {
    this.shortPathResolver = shortPathResolver;
  }

  @Override
  public int needsToNormalize(String path) {
    int n = path.length();
    int normalizationLevel = NORMALIZED;
    int dotCount = 0;
    char prevChar = 0;
    int segmentBeginIndex = 0; // The start index of the current path index
    boolean segmentHasShortPathChar = false; // Triggers more expensive short path regex test
    for (int i = 0; i < n; i++) {
      char c = path.charAt(i);
      if (isSeparator(c)) {
        if (c == '\\') {
          normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
        }
        // No need to check for '\\' here because that already causes normalization
        if (prevChar == '/') {
          normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
        }
        if (dotCount == 1 || dotCount == 2) {
          normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
        }
        if (segmentHasShortPathChar) {
          if (WindowsPathOperations.isShortPath(path.substring(segmentBeginIndex, i))) {
            normalizationLevel = Math.max(normalizationLevel, NEEDS_SHORT_PATH_NORMALIZATION);
          }
        }
        segmentBeginIndex = i + 1;
        segmentHasShortPathChar = false;
      } else if (c == '~') {
        // This path segment might be a Windows short path segment
        segmentHasShortPathChar = true;
      }
      dotCount = c == '.' ? dotCount + 1 : 0;
      prevChar = c;
    }
    if (segmentHasShortPathChar) {
      if (WindowsPathOperations.isShortPath(path.substring(segmentBeginIndex))) {
        normalizationLevel = Math.max(normalizationLevel, NEEDS_SHORT_PATH_NORMALIZATION);
      }
    }
    if ((n > 1 && isSeparator(prevChar)) || dotCount == 1 || dotCount == 2) {
      normalizationLevel = Math.max(normalizationLevel, NEEDS_NORMALIZE);
    }
    return normalizationLevel;
  }

  @Override
  public int needsToNormalizeSuffix(String normalizedSuffix) {
    // On Windows, all bets are off because of short paths, so we have to check the entire string
    return needsToNormalize(normalizedSuffix);
  }

  @Override
  public String normalize(String path, int normalizationLevel) {
    if (normalizationLevel == NORMALIZED) {
      return path;
    }
    if (normalizationLevel == NEEDS_SHORT_PATH_NORMALIZATION) {
      String resolvedPath = shortPathResolver.resolveShortPath(path);
      if (resolvedPath != null) {
        path = resolvedPath;
      }
    }
    String[] segments = Iterables.toArray(WINDOWS_PATH_SPLITTER.splitToList(path), String.class);
    int driveStrLength = getDriveStrLength(path);
    boolean isAbsolute = driveStrLength > 0;
    int segmentSkipCount = isAbsolute && driveStrLength > 1 ? 1 : 0;

    StringBuilder sb = new StringBuilder(path.length());
    if (isAbsolute) {
      char c = path.charAt(0);
      if (isSeparator(c)) {
        sb.append('/');
      } else {
        sb.append(Character.toUpperCase(c));
        sb.append(":/");
      }
    }
    int segmentCount = Utils.removeRelativePaths(segments, segmentSkipCount, isAbsolute);
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
    int n = path.length();
    if (n == 0) {
      return 0;
    }
    char c0 = path.charAt(0);
    if (isSeparator(c0)) {
      return 1;
    }
    if (n < 3) {
      return 0;
    }
    char c1 = path.charAt(1);
    char c2 = path.charAt(2);
    if (isDriveLetter(c0) && c1 == ':' && isSeparator(c2)) {
      return 3;
    }
    return 0;
  }

  private static boolean isDriveLetter(char c) {
    return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z'));
  }

  @Override
  public boolean isSeparator(char c) {
    return c == '/' || c == '\\';
  }

  @Override
  public char additionalSeparator() {
    return '\\';
  }

  @Override
  public String postProcessPathStringForExecution(String callablePathString) {
    // On Windows, .bat scripts (and possibly others) cannot be executed with forward slashes in
    // the path. Since backslashes are the standard path separator on Windows, we replace all
    // forward slashes with backslashes instead of trying to enumerate these special cases.
    return callablePathString.replace('/', '\\');
  }
}
