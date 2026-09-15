// Copyright 2022 The Bazel Authors. All rights reserved.
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

import javax.annotation.Nullable;

/** A batch of nodes requested from a {@link QueryableGraph}. */
@FunctionalInterface
public interface NodeBatch {

  /**
   * Returns the {@link NodeEntry} for the given key, or {@code null} if it does not exist.
   *
   * <p>Must only be called with a {@link SkyKey} that was part of the graph request for this batch,
   * otherwise behavior is undefined and may lead to incorrect evaluation results.
   */
  @Nullable
  NodeEntry get(SkyKey key);
}
