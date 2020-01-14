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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
import java.util.concurrent.atomic.AtomicReference;

/** Debugger API. */
// TODO(adonovan): move Debugger to Debug.Debugger.
public final class Debug {

  private Debug() {} // uninstantiable

  static final AtomicReference<Debugger> debugger = new AtomicReference<>();

  /**
   * Installs a global hook that causes subsequently executed Starlark threads to notify the
   * debugger of important events. Closes any previously set debugger. Call {@code
   * setDebugger(null)} to disable debugging.
   */
  public static void setDebugger(Debugger dbg) {
    Debugger prev = debugger.getAndSet(dbg);
    if (prev != null) {
      prev.close();
    }
  }

  /**
   * Returns a copy of the current stack of call frames, outermost call first.
   *
   * <p>This function is intended for use only when execution of {@code thread} is stopped, for
   * example at a breakpoint. The resulting DebugFrames should not be retained after execution of
   * the thread has resumed. Most clients should instead use {@link StarlarkThread#getCallStack}.
   */
  public static ImmutableList<Frame> getCallStack(StarlarkThread thread) {
    return thread.getDebugCallStack();
  }

  /** Debugger interface to the interpreter's internal call frame representation. */
  public interface Frame {

    /** Returns function called in this frame. */
    StarlarkCallable getFunction();

    /** Returns the location of the current program counter. */
    Location getLocation();

    /** Returns the local environment of this frame. */
    ImmutableMap<String, Object> getLocals();
  }
}
