// Copyright 2021 The Bazel Authors. All rights reserved.
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

import java.util.Collections;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * Lazily iterates over the segments of a path string.
 *
 * <p>Expects the path string to already be normalized.
 */
final class PathSegmentIterator implements Iterator<String> {

  static Iterator<String> create(String normalizedPath, int driveStrLength) {
    return normalizedPath.length() > driveStrLength
        ? new PathSegmentIterator(normalizedPath, driveStrLength)
        : Collections.emptyIterator();
  }

  private final String normalizedPath;
  private int start;

  private PathSegmentIterator(String normalizedPath, int driveStrLength) {
    this.normalizedPath = normalizedPath;
    this.start = driveStrLength;
  }

  @Override
  public boolean hasNext() {
    return start < normalizedPath.length();
  }

  @Override
  public String next() {
    if (!hasNext()) {
      throw new NoSuchElementException("No more segments: " + normalizedPath);
    }
    int end = start + 1;
    while (end < normalizedPath.length()
        && normalizedPath.charAt(end) != PathFragment.SEPARATOR_CHAR) {
      end++;
    }
    String segment = normalizedPath.substring(start, end);
    start = end + 1;
    return segment;
  }
}
