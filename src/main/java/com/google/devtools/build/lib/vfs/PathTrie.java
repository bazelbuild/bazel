// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.HashMap;
import java.util.Map;

/**
 * A trie that operates on path segments.
 *
 * @param <T> the type of the values.
 */
@ThreadCompatible
public class PathTrie<T> {
  @SuppressWarnings("unchecked")
  private static class Node<T> {
    private Node() {
      children = new HashMap<>();
    }

    private T value;
    private Map<String, Node<T>> children;
  }

  private final Node<T> root;

  public PathTrie() {
    root = new Node<T>();
  }

  /**
   * Puts a value in the trie.
   *
   * @param key must be an absolute path.
   */
  public void put(PathFragment key, T value) {
    Preconditions.checkArgument(key.isAbsolute(), "PathTrie only accepts absolute paths as keys.");
    Node<T> current = root;
    for (String segment : key.getSegments()) {
      current.children.putIfAbsent(segment, new Node<T>());
      current = current.children.get(segment);
    }
    current.value = value;
  }

  /**
   * Gets a value from the trie. If there is an entry with the same key, that will be returned,
   * otherwise, the value corresponding to the key that matches the longest prefix of the input.
   */
  public T get(PathFragment key) {
    Node<T> current = root;
    T lastValue = current.value;

    for (String segment : key.getSegments()) {
      if (current.children.containsKey(segment)) {
        current = current.children.get(segment);
        // Track the values of increasing matching prefixes.
        if (current.value != null) {
          lastValue = current.value;
        }
      } else {
        // We've reached the longest prefix, no further to go.
        break;
      }
    }

    return lastValue;
  }
}
