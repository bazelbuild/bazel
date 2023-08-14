// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.util;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Stores a collection of paths as a trie safe for concurrent read/write access.
 *
 * <p>All paths must be relative and non-empty. Once a path has been added it cannot be removed.
 */
public final class ConcurrentPathTrie {

  private enum Match {
    NONE,
    EXACT,
    PREFIX
  }

  private static final class Node {
    final ConcurrentHashMap<String, Node> tab = new ConcurrentHashMap<>();
    Match state = Match.NONE;
  }

  private final Node root = new Node();

  /** Returns whether the trie contains the given path. */
  public boolean contains(PathFragment path) {
    checkValid(path);

    Node node = root;
    Iterator<String> segmentIterator = path.segments().iterator();

    while (segmentIterator.hasNext()) {
      String segment = segmentIterator.next();
      boolean isLast = !segmentIterator.hasNext();

      node = node.tab.get(segment);

      if (node == null) {
        // No match.
        return false;
      }

      if (node.state == Match.PREFIX) {
        // Prefix match.
        return true;
      }

      if (isLast && node.state == Match.EXACT) {
        // Exact match.
        return true;
      }
    }

    // Reached end of input path without a match.
    return false;
  }

  /** Adds the given path to the trie. */
  public void add(PathFragment path) {
    addInternal(path, false);
  }

  /** Adds the set of paths with the given prefix to the trie. */
  public void addPrefix(PathFragment path) {
    addInternal(path, true);
  }

  private void addInternal(PathFragment path, boolean asPrefix) {
    checkValid(path);

    Node node = root;
    Iterator<String> segmentIterator = path.segments().iterator();

    while (segmentIterator.hasNext()) {
      String segment = segmentIterator.next();
      boolean isLast = !segmentIterator.hasNext();

      node =
          node.tab.compute(
              segment,
              (unusedKey, val) -> {
                if (val == null) {
                  val = new Node();
                }
                if (isLast) {
                  if (asPrefix) {
                    // Add as prefix match.
                    val.state = Match.PREFIX;
                  } else {
                    // Add as exact match.
                    val.state = Match.EXACT;
                  }
                }
                return val;
              });
    }
  }

  private static void checkValid(PathFragment path) {
    checkArgument(
        !path.isAbsolute() && !path.isEmpty(),
        "ConcurrentPathTrie only accepts relative non-empty paths");
  }
}
