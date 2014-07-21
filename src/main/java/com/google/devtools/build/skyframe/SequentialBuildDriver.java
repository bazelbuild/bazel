// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.events.ErrorEventListener;

/**
 * A driver for an auto-updating graph which operates over monotonically increasing integer
 * versions.
 */
public class SequentialBuildDriver {
  private final AutoUpdatingGraph autoUpdatingGraph;
  private IntVersion curVersion;

  public SequentialBuildDriver(AutoUpdatingGraph graph) {
    this.autoUpdatingGraph = Preconditions.checkNotNull(graph);
    this.curVersion = new IntVersion(0);
  }

  public <T extends Node> UpdateResult<T> update(Iterable<NodeKey> roots, boolean keepGoing,
                                                 int numThreads, ErrorEventListener reporter)
      throws InterruptedException {
    try {
      return autoUpdatingGraph.update(roots, curVersion, keepGoing, numThreads, reporter);
    } finally {
      curVersion = curVersion.next();
    }
  }

  public AutoUpdatingGraph getGraph() {
    return autoUpdatingGraph;
  }
}
