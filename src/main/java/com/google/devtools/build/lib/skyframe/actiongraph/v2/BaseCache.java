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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Basic class to abstract action graph cache functionality.
 */
abstract class BaseCache<K, P> {
  private final Map<K, Integer> cache = new HashMap<>();
  protected final AqueryOutputHandler aqueryOutputHandler;

  BaseCache(AqueryOutputHandler aqueryOutputHandler) {
    this.aqueryOutputHandler = aqueryOutputHandler;
  }

  private int generateNextId() {
    // protobuf interprets the value 0 as "default value" for uint64, thus treating the field as
    // "unset". We should start from 1 instead.
    return cache.size() + 1;
  }

  protected K transformToKey(K data) {
    // In most cases, the data is the key but it can be overridden by subclasses.
    return data;
  }

  /**
   * Store the data in the internal cache, if it's not yet present. Return the generated id. Ids are
   * positive and unique.
   *
   * <p>Stream the proto to output, the first time it's generated.
   */
  int dataToIdAndStreamOutputProto(K data) throws IOException, InterruptedException {
    K key = transformToKey(data);
    Integer id = cache.get(key);
    if (id == null) {
      // Note that this cannot be replaced by computeIfAbsent since createProto is a recursive
      // operation for the case of nested sets which will call dataToId on the same object and thus
      // computeIfAbsent again.
      id = generateNextId();
      cache.put(key, id);
      P proto = createProto(data, id);
      toOutput(proto);
    }
    return id;
  }

  abstract P createProto(K key, int id) throws IOException, InterruptedException;

  abstract void toOutput(P proto) throws IOException;
}
