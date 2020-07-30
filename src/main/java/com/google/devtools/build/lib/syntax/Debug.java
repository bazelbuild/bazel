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
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Debugger API. */
// TODO(adonovan): move Debugger to Debug.Debugger.
public final class Debug {

  /** A Starlark value that can expose additional information to a debugger. */
  public interface ValueWithDebugAttributes extends StarlarkValue {
    /**
     * Returns a list of DebugAttribute of this value. For example, it can be the internal fields of
     * a value that are not accessible from Starlark, or the values inside a collection.
     */
    ImmutableList<DebugAttribute> getDebugAttributes();
  }

  /** A name/value pair used in the return value of getDebugAttributes. */
  public static final class DebugAttribute {
    public final String name;
    public final Object value; // a legal Starlark value

    public DebugAttribute(String name, Object value) {
      this.name = name;
      this.value = value;
    }
  }

  /** See stepControl */
  public interface ReadyToPause extends Predicate<StarlarkThread> {}

  /**
   * Describes the stepping behavior that should occur when execution of a thread is continued.
   * (Debugger API)
   */
  public enum Stepping {
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

  /**
   * Given a requested stepping behavior, returns a predicate over the context that tells the
   * debugger when to pause. (Debugger API)
   *
   * <p>The predicate will return true if we are at the next statement where execution should pause,
   * and it will return false if we are not yet at that statement. No guarantee is made about the
   * predicate's return value after we have reached the desired statement.
   *
   * <p>A null return value indicates that no further pausing should occur.
   */
  @Nullable
  public static Debug.ReadyToPause stepControl(StarlarkThread th, Debug.Stepping stepping) {
    final int depth = th.getCallStackSize();
    switch (stepping) {
      case NONE:
        return null;
      case INTO:
        // pause at the very next statement
        return thread -> true;
      case OVER:
        return thread -> thread.getCallStackSize() <= depth;
      case OUT:
        // if we're at the outermost frame, same as NONE
        return depth == 0 ? null : thread -> thread.getCallStackSize() < depth;
    }
    throw new IllegalArgumentException("Unsupported stepping type: " + stepping);
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

  /**
   * Interface by which debugging tools are notified of a thread entering or leaving its top-level
   * frame.
   */
  public interface ThreadHook {
    void onPushFirst(StarlarkThread thread);

    void onPopLast(StarlarkThread thread);
  }

  static ThreadHook threadHook = null;

  /**
   * Installs a global hook that is notified each time a thread pushes or pops its top-level frame.
   * This interface is provided to support special tools; ordinary clients should have no need for
   * it.
   */
  public static void setThreadHook(ThreadHook hook) {
    threadHook = hook;
  }
}
