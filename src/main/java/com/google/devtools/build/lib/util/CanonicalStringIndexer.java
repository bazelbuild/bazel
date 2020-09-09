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
package com.google.devtools.build.lib.util;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.Map;

/**
 * A string indexer backed by a map and reverse index lookup.
 * Every unique string is stored in memory exactly once.
 */
@ThreadSafe
public class CanonicalStringIndexer extends AbstractIndexer {

  private static final int NOT_FOUND = -1;

  // This is similar to (Synchronized) BiMap.
  // These maps *must* be weakly threadsafe to ensure thread safety for string
  // indexer as a whole. Specifically, mutating operations are serialized, but
  // read-only operations may be executed concurrently with mutators.
  private final Map<String, Integer> stringToInt;
  private final Map<Integer, String> intToString;

  /*
   * Creates an indexer instance from two backing maps. These maps may be
   * pre-initialized with data, but *must*:
   * a. Support read-only operations done concurrently with mutations.
   *    Note that mutations will be serialized.
   * b. Be reverse mappings of each other, if pre-initialized.
   */
  public CanonicalStringIndexer(Map<String, Integer> stringToInt,
                                Map<Integer, String> intToString) {
    Preconditions.checkArgument(stringToInt.size() == intToString.size());
    this.stringToInt = stringToInt;
    this.intToString = intToString;
  }


  @Override
  public synchronized void clear() {
    stringToInt.clear();
    intToString.clear();
  }

  @Override
  public int size() {
    return intToString.size();
  }

  @Override
  public int getOrCreateIndex(String s) {
    Integer i = stringToInt.get(s);
    if (i == null) {
      s = StringCanonicalizer.intern(s);
      synchronized (this) {
        // First, make sure another thread hasn't just added the entry:
        i = stringToInt.get(s);
        if (i != null) {
          return i;
        }

        int ind = intToString.size();
        stringToInt.put(s, ind);
        intToString.put(ind, s);
        return ind;
      }
    } else {
      return i;
    }
  }

  @Override
  public int getIndex(String s) {
    Integer i = stringToInt.get(s);
    return (i == null) ? NOT_FOUND : i;
  }

  @Override
  public synchronized boolean addString(String s) {
    int originalSize = size();
    getOrCreateIndex(s);
    return (size() > originalSize);
  }

  @Override
  public String getStringForIndex(int i) {
    return intToString.get(i);
  }

  @Override
  public synchronized String toString() {
    StringBuilder builder = new StringBuilder();
    builder.append("size = ").append(size()).append("\n");
    for (Map.Entry<String, Integer> entry : stringToInt.entrySet()) {
      builder.append(entry.getKey()).append(" <==> ").append(entry.getValue()).append("\n");
    }
    return builder.toString();
  }
}
