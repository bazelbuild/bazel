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
package com.google.devtools.build.android.dexer;

import com.google.common.annotations.VisibleForTesting;
import java.util.Comparator;
import java.util.zip.ZipEntry;

/**
 * Comparator that orders {@link ZipEntry ZipEntries} {@link #LIKE_DX like Android's dx tool}.
 */
enum ZipEntryComparator implements Comparator<ZipEntry> {
  /**
   * Comparator to order more or less order alphabetically by file name.  See
   * {@link #compareClassNames} for the exact name comparison.
   */
  LIKE_DX;

  @Override
  // Copied from com.android.dx.cf.direct.ClassPathOpener
  public int compare(ZipEntry a, ZipEntry b) {
    return compareClassNames(a.getName(), b.getName());
  }

  /**
   * Sorts java class names such that outer classes precede their inner classes and "package-info"
   * precedes all other classes in its package.
   *
   * @param a {@code non-null;} first class name
   * @param b {@code non-null;} second class name
   * @return {@code compareTo()}-style result
   */
  // Copied from com.android.dx.cf.direct.ClassPathOpener
  @VisibleForTesting
  static int compareClassNames(String a, String b) {
    // Ensure inner classes sort second
    a = a.replace('$', '0');
    b = b.replace('$', '0');

    /*
     * Assuming "package-info" only occurs at the end, ensures package-info
     * sorts first.
     */
    a = a.replace("package-info", "");
    b = b.replace("package-info", "");

    return a.compareTo(b);
  }
}
