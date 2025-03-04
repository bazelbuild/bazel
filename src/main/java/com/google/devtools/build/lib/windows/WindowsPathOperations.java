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
package com.google.devtools.build.lib.windows;

import com.google.devtools.build.lib.jni.JniLoader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Support functions for Windows short paths (eg. "C:/progra~1") */
public final class WindowsPathOperations {

  private WindowsPathOperations() {}

  static {
    JniLoader.loadJni();
  }

  // Properties of 8dot3 (DOS-style) short file names:
  // - they are at most 11 characters long
  // - they have a prefix (before "~") that is {1..6} characters long, may contain numbers, letters,
  //   "_", even "~", and maybe even more
  // - they have a "~" after the prefix
  // - have {1..6} numbers after "~" (according to [1] this is only one digit, but MSDN doesn't
  //   clarify this), the combined length up till this point is at most 8
  // - they have an optional "." afterwards, and another {0..3} more characters
  // - just because a path looks like a short name it isn't necessarily one; the user may create
  //   such names and they'd resolve to themselves
  // [1] https://en.wikipedia.org/wiki/8.3_filename#VFAT_and_Computer-generated_8.3_filenames
  //     bullet point (3) (on 2016-12-05)
  private static final Pattern PATTERN = Pattern.compile("^(.{1,6})~([0-9]{1,6})(\\..{0,3}){0,1}");

  /** Matches a single path segment for whether it could be a Windows short path. */
  public static boolean isShortPath(String segment) {
    Matcher m = PATTERN.matcher(segment);
    return segment.length() <= 12
        && m.matches()
        && m.groupCount() >= 2
        && (m.group(1).length() + m.group(2).length()) < 8; // the "~" makes it at most 8
  }

  /**
   * Returns the long path associated with the input `path`.
   *
   * <p>This method resolves all 8dot3 style components of the path and returns the long format. For
   * example, if the input is "C:/progra~1/micros~1" the result may be "C:\Program Files\Microsoft
   * Visual Studio 14.0". The returned path is Windows-style in that it uses backslashes, even if
   * the input uses forward slashes.
   *
   * <p>May return an UNC path if `path` or its resolution is sufficiently long.
   *
   * @throws IOException if the `path` is not found or some other I/O error occurs
   */
  public static String getLongPath(String path) throws IOException {
    String[] result = new String[] {null};
    String[] error = new String[] {null};
    if (nativeGetLongPath(asLongPath(path), result, error)) {
      return removeUncPrefixAndUseSlashes(result[0]);
    } else {
      throw new IOException(error[0]);
    }
  }

  /** Returns a Windows-style path suitable to pass to unicode WinAPI functions. */
  static String asLongPath(String path) {
    return !path.startsWith("\\\\?\\")
        ? ("\\\\?\\" + path.replace('/', '\\'))
        : path.replace('/', '\\');
  }

  static String removeUncPrefixAndUseSlashes(String p) {
    if (p.length() >= 4
        && p.charAt(0) == '\\'
        && (p.charAt(1) == '\\' || p.charAt(1) == '?')
        && p.charAt(2) == '?'
        && p.charAt(3) == '\\') {
      p = p.substring(4);
    }
    return p.replace('\\', '/');
  }

  private static native boolean nativeGetLongPath(String path, String[] result, String[] error);
}
