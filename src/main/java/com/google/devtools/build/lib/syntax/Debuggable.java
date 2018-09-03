// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.events.Location;
import java.util.Collection;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** A context in which debugging can occur. Implemented by Skylark environments. */
public interface Debuggable {

  /** Evaluates a Skylark statement in the adapter's environment. */
  Object evaluate(String statement) throws EvalException, InterruptedException;

  /**
   * Returns the stack frames corresponding of the context's current (paused) state.
   *
   * <p>For all stack frames except the innermost, location information is retrieved from the
   * current context. The innermost frame's location must be supplied as {@code currentLocation} by
   * the caller.
   */
  Collection<DebugFrame> listFrames(Location currentLocation);

  /**
   * Given a requested stepping behavior, returns a predicate over the context that tells the
   * debugger when to pause.
   *
   * <p>The predicate will return true if we are at the next statement where execution should pause,
   * and it will return false if we are not yet at that statement. No guarantee is made about the
   * predicate's return value after we have reached the desired statement.
   *
   * <p>A null return value indicates that no further pausing should occur.
   */
  @Nullable
  ReadyToPause stepControl(Stepping stepping);

  /**
   * When stepping, this determines whether or not the context has yet reached a state for which
   * execution should be paused.
   *
   * <p>A single instance is only useful for advancing by one pause. A new instance may be required
   * after that.
   */
  interface ReadyToPause extends Predicate<Environment> {}

  /** Describes the stepping behavior that should occur when execution of a thread is continued. */
  enum Stepping {
    /** Continue execution without stepping. */
    NONE,
    /**
     * If the thread is paused on a statement that contains a function call, step into that
     * function. Otherwise, this is the same as OVER.
     */
    INTO,
    /**
     * Step over the current statement and any functions that it may call, stopping at the next
     * statement in the same frame. If no more statements are available in the current frame, same
     * as OUT.
     */
    OVER,
    /**
     * Continue execution until the current frame has been exited and then pause. If we are
     * currently in the outer-most frame, same as NONE.
     */
    OUT,
  }
}
