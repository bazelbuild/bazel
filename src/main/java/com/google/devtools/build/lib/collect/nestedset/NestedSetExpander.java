// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.collect.ImmutableList;

/**
 * Helper class to expand {@link NestedSet} instances.
 *
 * <p>Implementations besides {@link NO_CALLBACKS} may wish to implement callbacks or timeouts for
 * dealing with expansions of sets from storage.
 */
public interface NestedSetExpander {

  <T> ImmutableList<? extends T> toListInterruptibly(NestedSet<? extends T> nestedSet)
      throws InterruptedException;

  /** Simply delegates to {@link NestedSet#toListInterruptibly} without doing anything special. */
  NestedSetExpander DEFAULT = NestedSet::toListInterruptibly;
}
