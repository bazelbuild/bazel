// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkdebug.server;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;

/**
 * A map of identifier to value for all the values with children which have been communicated to the
 * debug client for a particular, currently-paused thread.
 *
 * <p>Objects without children (as defined by the debugging protocol) should not be stored here.
 *
 * <p>Object identifiers apply only to a single thread, and aren't relevant to other threads.
 *
 * <p>This state is retained only while the thread is paused.
 */
final class ThreadObjectMap {
  private final AtomicLong lastId = new AtomicLong(1);
  private final Map<Long, Object> objectMap = new ConcurrentHashMap<>();

  /**
   * Assigns and returns a unique identifier for this object, and stores a reference in the
   * identifier to object map.
   *
   * <p>If called multiple times for a given object, a different identifier will be returned each
   * time. The object can be retrieved by all such identifiers.
   */
  long registerValue(Object value) {
    long id = lastId.getAndIncrement();
    objectMap.put(id, value);
    return id;
  }

  /** Returns the object corresponding to the given identifier, if one exists. */
  @Nullable
  Object getValue(long identifier) {
    return objectMap.get(identifier);
  }
}
