package net.starlark.java.eval;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Hold the information about the number of debuggers or profilers
 * attached to this process.
 */
class DebugProfile {

  private static final AtomicInteger numberOfAgents = new AtomicInteger(0);

  /** Add or subtract the number of agents (profilers or debuggers). */
  static void add(int delta) {
    numberOfAgents.addAndGet(delta);
  }

  /** True iff the number of agents (profilers or debuggers) is non-zero. */
  static boolean hasAny() {
    return numberOfAgents.get() != 0;
  }
}
