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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.skyframe.InMemoryGraphImpl.EdgelessInMemoryGraphImpl;

/** Tests for {@link InMemoryGraphImpl}. */
public class InMemoryGraphTest extends GraphTest {

  @Override
  protected Version getStartingVersion() {
    return IntVersion.of(0);
  }

  @Override
  protected Version getNextVersion(Version v) {
    Preconditions.checkState(v instanceof IntVersion);
    return ((IntVersion) v).next();
  }

  @Override
  protected void makeGraph() {
    graph = new InMemoryGraphImpl();
  }

  @Override
  protected ProcessableGraph getGraph(Version version) {
    return graph;
  }

  /** Tests for {@link EdgelessInMemoryGraphImpl}. */
  public static final class EdgelessInMemoryGraphTest extends InMemoryGraphTest {

    @Override
    protected void makeGraph() {}

    @Override
    protected ProcessableGraph getGraph(Version version) {
      return new EdgelessInMemoryGraphImpl();
    }

    @Override
    protected Version getNextVersion(Version version) {
      throw new UnsupportedOperationException();
    }

    @Override
    protected boolean shouldTestIncrementality() {
      return false;
    }
  }
}
