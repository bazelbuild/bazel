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

package com.google.devtools.build.lib.events;

import javax.annotation.Nullable;

/** An event that can be reported to an {@link ExtendedEventHandler}. */
public interface Reportable {

  void reportTo(ExtendedEventHandler handler);

  /**
   * If this event supports tag-based output filtering, returns a new instance identical to this one
   * but with the given tag. Otherwise returns {@code this}.
   *
   * <p>Tags can be used to apply filtering to events. See {@link OutputFilter}.
   */
  Reportable withTag(@Nullable String tag);

  /**
   * If this event originated from {@link
   * com.google.devtools.build.skyframe.SkyFunction.Environment#getListener}, whether it should be
   * stored in the corresponding Skyframe node to be replayed on incremental builds when the node is
   * deemed up-to-date.
   *
   * <p>Events which are crucial to the correctness of the evaluation should return {@code true} so
   * that they are replayed when the {@link com.google.devtools.build.skyframe.SkyFunction}
   * invocation is cached. On the other hand, events that are merely informational (such as a
   * progress update) should return {@code false} to avoid taking up memory.
   *
   * <p>Evaluations may disable all event storage and replay by using a custom {@link
   * com.google.devtools.build.skyframe.EventFilter}, in which case this method is only used to
   * fulfill the semantics described at {@link
   * com.google.devtools.build.skyframe.SkyFunction.Environment#getListener}.
   *
   * <p>This method is not relevant for events which do not originate from {@link
   * com.google.devtools.build.skyframe.SkyFunction} evaluation.
   *
   * <p>Classes returning {@code true} should have cheap {@link Object#hashCode()} and {@link
   * Object#equals(Object)} implementations.
   */
  default boolean storeForReplay() {
    return false;
  }
}
