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

import com.google.devtools.build.lib.concurrent.ThreadSafety;

/**
 * Receiver to inform callers which nodes have been invalidated. Nodes may be invalidated and then
 * re-validated if they have been found not to be changed.
 */
public interface NodeProgressReceiver {
  /**
   * New state of the node entry after evaluation.
   */
  enum EvaluationState {
    /** The node was successfully re-evaluated. */
    BUILT,
    /** The node is clean or re-validated. */
    CLEAN,
  }

  /**
   * New state of the node entry after invalidation.
   */
  enum InvalidationState {
    /** The node is dirty, although it might get re-validated again. */
    DIRTY,
    /** The node is dirty and got deleted, cannot get re-validated again. */
    DELETED,
  }

  /**
   * Notifies that {@code node} has been invalidated.
   *
   * <p>{@code state} indicates the new state of the node.
   *
   * <p>This method is not called on invalidation of nodes which do not have a value (usually
   * because they are in error).
   *
   * <p>May be called concurrently from multiple threads, possibly with the same {@code node}
   * object.
   */
  @ThreadSafety.ThreadSafe
  void invalidated(Node node, InvalidationState state);

  /**
   * Notifies that {@code nodeKey} is about to get queued for evaluation.
   *
   * <p>Note that we don't guarantee that it actually got enqueued or will, only that if
   * everything "goes well" (e.g. no interrupts happen) it will.
   *
   * <p>This guarantee is intentionally vague to encourage writing robust implementations.
   */
  @ThreadSafety.ThreadSafe
  void enqueueing(NodeKey nodeKey);

  /**
   * Notifies that {@code node} has been evaluated.
   *
   * <p>{@code state} indicates the new state of the node.
   *
   * <p>This method is not called if the node builder threw an error when building this node.
   */
  @ThreadSafety.ThreadSafe
  void evaluated(NodeKey nodeKey, Node node, EvaluationState state);
}
