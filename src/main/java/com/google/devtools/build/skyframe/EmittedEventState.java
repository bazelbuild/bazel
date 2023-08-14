// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import java.util.Set;

/**
 * Keeps track of visited nodes for {@linkplain com.google.devtools.build.lib.events.Reportable
 * events} stored in a {@link com.google.devtools.build.lib.collect.nestedset.NestedSet}.
 *
 * <p>Also tracks warnings for purposes of deduplication, since those are not {@linkplain
 * Event#storeForReplay stored}.
 */
public final class EmittedEventState implements NestedSetVisitor.VisitedState {

  private final Set<Object> seenNodes = Sets.newConcurrentHashSet();
  private final Set<Event> seenWarnings = Sets.newConcurrentHashSet();

  /** Clears the seen nodes and warnings. */
  public void clear() {
    seenNodes.clear();
    seenWarnings.clear();
  }

  @Override
  public boolean add(Object node) {
    return seenNodes.add(node);
  }

  /** Returns {@code true} if the given warning was not seen before. */
  boolean addWarning(Event warning) {
    checkArgument(warning.getKind() == EventKind.WARNING, "Not a warning: %s", warning);
    return seenWarnings.add(warning);
  }
}
