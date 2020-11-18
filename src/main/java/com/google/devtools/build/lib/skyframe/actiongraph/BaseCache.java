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
package com.google.devtools.build.lib.skyframe.actiongraph;

import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import java.util.HashMap;
import java.util.Map;

/**
 * Basic class to abstract action graph cache functionality.
 */
abstract class BaseCache<K, P> {
  private final Map<K, String> cache = new HashMap<>();
  protected final ActionGraphContainer.Builder actionGraphBuilder;

  BaseCache(ActionGraphContainer.Builder actionGraphBuilder) {
    this.actionGraphBuilder = actionGraphBuilder;
  }

  private String generateNextId() {
    return String.valueOf(cache.size());
  }

  protected K transformToKey(K data) {
    // In most cases, the data is the key but it can be overridden by subclasses.
    return data;
  }

  String dataToId(K data) throws InterruptedException {
    K key = transformToKey(data);
    String id = cache.get(key);
    if (id == null) {
      // Note that this cannot be replaced by computeIfAbsent since createProto is a recursive
      // operation for the case of nested sets which will call dataToId on the same object and thus
      // computeIfAbsent again.
      id = generateNextId();
      cache.put(key, id);
      P proto = createProto(data, id);
      addToActionGraphBuilder(proto);
    }
    return id;
  }

  abstract P createProto(K key, String id) throws InterruptedException;

  abstract void addToActionGraphBuilder(P proto);
}
