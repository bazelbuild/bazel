// Copyright 2016 The Bazel Authors. All rights reserved.
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

/** Filters out events which should not be stored during evaluation in {@link ParallelEvaluator}. */
public interface EventFilter {

  /**
   * Returns true if any {@linkplain com.google.devtools.build.lib.events.Reportable events} should
   * be stored in skyframe nodes. Otherwise, optimizations may be made to avoid doing unnecessary
   * work when evaluating node entries.
   */
  boolean storeEvents();

  /**
   * Determines whether stored {@linkplain com.google.devtools.build.lib.events.Reportable events}
   * should propagate from {@code depKey} to {@code primaryKey}.
   *
   * <p>Only relevant if {@link #storeEvents} returns {@code true}.
   */
  boolean shouldPropagate(SkyKey depKey, SkyKey primaryKey);

  EventFilter FULL_STORAGE =
      new EventFilter() {
        @Override
        public boolean storeEvents() {
          return true;
        }

        @Override
        public boolean shouldPropagate(SkyKey depKey, SkyKey primaryKey) {
          return true;
        }
      };

  EventFilter NO_STORAGE =
      new EventFilter() {
        @Override
        public boolean storeEvents() {
          return false;
        }

        @Override
        public boolean shouldPropagate(SkyKey depKey, SkyKey primaryKey) {
          throw new UnsupportedOperationException();
        }
      };
}
