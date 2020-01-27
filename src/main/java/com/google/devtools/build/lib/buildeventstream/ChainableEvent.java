// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream;

import java.util.Collection;

/**
 * Interface for objects organised in a DAG-like fashion.
 *
 * <p>When objects that are naturally organised in a directed acyclic graph are sent sequentially
 * over some channel, the graph-structure needs to represented in some way. We chose the
 * representation that each node knows its immediate successor nodes (rather than its predecessors)
 * to suit the * properties of the event stream: new parents of a node might be discovered late,
 * e.g., if a test suite is only expanded after one of its tests is already finished as it was also
 * needed for another target.
 */
public interface ChainableEvent {

  /**
   * Provide the identifier of the event.
   *
   * <p>Event identifiers have to be unique within the set of events belonging to the same build
   * invocation.
   */
  BuildEventId getEventId();

  /**
   * Provide the children of the event.
   *
   * <p>It is a requirement of a well-formed event stream that for every event that does not
   * indicate the beginning of a new build, at least one parent be present before the event itself.
   * However, more parents might appear later in the stream (e.g., if a test suite expanded later
   * discovers that a test that is already completed belongs to it).
   *
   * <p>A build-event stream is finished if and only if all announced children have occurred.
   */
  Collection<BuildEventId> getChildrenEvents();
}
