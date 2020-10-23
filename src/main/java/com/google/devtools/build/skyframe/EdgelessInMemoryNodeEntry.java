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

/**
 * NodeEntry that does not store edges (directDeps and reverseDeps) when the node is done. Used to
 * save memory when it is known that the graph will not be reused.
 *
 * <p>Graph edges must be stored for incremental builds, but if this program will terminate after a
 * single run, edges can be thrown away in order to save memory. The edges will be stored in the
 * {@link BuildingState} as usual while the node is being built, but will not be stored once the
 * node is done and written to the graph. Any attempt to access the edges once the node is done will
 * fail the build fast.
 */
public class EdgelessInMemoryNodeEntry extends InMemoryNodeEntry {
  @Override
  public KeepEdgesPolicy keepEdges() {
    return KeepEdgesPolicy.NONE;
  }

  @Override
  protected void postProcessAfterDone() {
    this.directDeps = null;
    this.reverseDeps = null;
  }
}
