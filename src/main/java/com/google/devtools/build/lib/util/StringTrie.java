// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.util;

import com.google.common.base.Preconditions;


/**
 * A trie that operates on path segments of an input string instead of individual characters.
 *
 * <p>Only accepts strings that contain only low-ASCII characters (0-127)
 *
 * @param <T> the type of the values
 */
public class StringTrie<T> {
  private static final int RANGE = 128;

  @SuppressWarnings("unchecked")
  private static class Node<T> {
    private Node() {
      children = new Node[RANGE];
    }

    private T value;
    private Node<T> children[];
  }

  private final Node<T> root;

  public StringTrie() {
    root = new Node<T>();
  }

  /**
   * Puts a value in the trie.
   */
  public void put(CharSequence key, T value) {
    Node<T> current = root;

    for (int i = 0; i < key.length(); i++) {
      char ch = key.charAt(i);
      Preconditions.checkState(ch < RANGE);
      Node<T> next = current.children[ch];
      if (next == null) {
        next = new Node<T>();
        current.children[ch] = next;
      }

      current = next;
    }

    current.value = value;
  }

  /**
   * Gets a value from the trie. If there is an entry with the same key, that will be returned,
   * otherwise, the value corresponding to the key that matches the longest prefix of the input.
   */
  public T get(String key) {
    Node<T> current = root;
    T lastValue = current.value;

    for (int i = 0; i < key.length(); i++) {
      char ch = key.charAt(i);
      Preconditions.checkState(ch < RANGE);

      current = current.children[ch];
      if (current == null) {
        break;
      }

      if (current.value != null) {
        lastValue = current.value;
      }
    }

    return lastValue;
  }
}
