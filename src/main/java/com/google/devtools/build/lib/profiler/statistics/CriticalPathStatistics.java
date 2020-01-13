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
package com.google.devtools.build.lib.profiler.statistics;

import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo;
import com.google.devtools.build.lib.profiler.analysis.ProfileInfo.CriticalPathEntry;
import java.util.Iterator;

/**
 * Keeps a predefined list of {@link CriticalPathEntry}'s cumulative durations and allows iterating
 * over pairs of their descriptions and relative durations.
 */
public final class CriticalPathStatistics {
  /**
   * The actual critical path.
   */
  private final CriticalPathEntry totalPath;

  public CriticalPathStatistics(ProfileInfo info) {
    totalPath = info.getCriticalPath();
  }

  /**
   * @return the critical path obtained by not filtering out any {@link ProfilerTask}
   */
  public CriticalPathEntry getTotalPath() {
    return totalPath;
  }

  /**
   * Constructs a filtered Iterable from a critical path.
   *
   * <p>Ignores all fake (task id < 0) path entries.
   */
  public Iterable<CriticalPathEntry> getFilteredPath(final CriticalPathEntry path) {
    return () ->
        new Iterator<CriticalPathEntry>() {
          private CriticalPathEntry nextEntry = path;

          @Override
          public boolean hasNext() {
            return nextEntry != null;
          }

          @Override
          public CriticalPathEntry next() {
            CriticalPathEntry current = nextEntry;
            do {
              nextEntry = nextEntry.next;
            } while (nextEntry != null && nextEntry.task.isFake());
            return current;
          }

          @Override
          public void remove() {
            throw new UnsupportedOperationException();
          }
        };
  }
}

