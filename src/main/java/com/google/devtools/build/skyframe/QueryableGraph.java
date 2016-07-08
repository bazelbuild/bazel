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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.util.EnumSet;
import java.util.Map;

import javax.annotation.Nullable;

/** A graph that exposes its entries and structure, for use by classes that must traverse it. */
@ThreadSafe
public interface QueryableGraph {
  /** Returns the node with the given name, or {@code null} if the node does not exist. */
  @Nullable
  NodeEntry get(SkyKey key);

  /**
   * Fetches all the given nodes. Returns a map {@code m} such that, for all {@code k} in {@code
   * keys}, {@code m.get(k).equals(e)} iff {@code get(k) == e} and {@code e != null}, and {@code
   * !m.containsKey(k)} iff {@code get(k) == null}. The {@code fields} parameter is a hint to the
   * QueryableGraph implementation that allows it to possibly construct certain fields of the
   * returned node entries more lazily. Hints may only be applied to nodes in a certain state, like
   * done nodes.
   */
  Map<SkyKey, NodeEntry> getBatchWithFieldHints(
      Iterable<SkyKey> keys, EnumSet<NodeEntryField> fields);
}
