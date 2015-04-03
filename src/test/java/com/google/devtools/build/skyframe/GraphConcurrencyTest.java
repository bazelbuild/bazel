// Copyright 2015 Google Inc. All rights reserved.
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

import org.junit.Before;
import org.junit.Test;

/** Base class for concurrency sanity tests on {@link EvaluableGraph} implementations. */
public abstract class GraphConcurrencyTest {

  private static final SkyFunctionName SKY_FUNCTION_NAME =
      new SkyFunctionName("GraphConcurrencyTestKey", /*isComputed=*/false);
  private ProcessableGraph graph;

  protected abstract ProcessableGraph getGraph();

  @Before
  public void init() {
    this.graph = getGraph();
  }

  private SkyKey key(String name) {
    return new SkyKey(SKY_FUNCTION_NAME, name);
  }

  @Test
  public void createIfAbsentSanity() {
    graph.createIfAbsent(key("cat"));
  }

  // TODO(bazel-team): Add tests.
}
