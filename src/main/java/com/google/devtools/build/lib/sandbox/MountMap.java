// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ForwardingSortedMap;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * A map that throws an exception when trying to replace a key (i.e. once a key gets a value,
 * any additional attempt of putting a value on the same key will throw an exception).
 *
 * <p>Returns entries sorted by path depth (shorter paths first) and in lexicographical order.
 */
final class MountMap extends ForwardingSortedMap<Path, Path> {
  final TreeMap<Path, Path> delegate = new TreeMap<>();

  @Override
  protected SortedMap<Path, Path> delegate() {
    return delegate;
  }

  @Override
  public Path put(Path key, Path value) {
    Path previousValue = get(key);
    if (previousValue == null) {
      return super.put(key, value);
    } else if (previousValue.equals(value)) {
      return value;
    } else {
      throw new IllegalArgumentException(
          String.format("Cannot mount both '%s' and '%s' onto '%s'", previousValue, value, key));
    }
  }

  @Override
  public void putAll(Map<? extends Path, ? extends Path> map) {
    for (Entry<? extends Path, ? extends Path> entry : map.entrySet()) {
      put(entry.getKey(), entry.getValue());
    }
  }
}
