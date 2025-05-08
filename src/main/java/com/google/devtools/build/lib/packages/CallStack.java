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
package com.google.devtools.build.lib.packages;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.HashCodes;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Creates compact representations of Starlark call stacks for rule instantiations.
 *
 * <p>Implementation is optimized for minimizing memory overhead by sharing {@link Node} instances
 * when two call stacks have a common tail. For example, two different BUILD files that call into
 * the same macro can share {@link Node} instances.
 *
 * <p>The sharing rate of interior nodes is expected to be high, so nodes are implemented as a
 * linked list to eliminate array cost.
 */
final class CallStack {

  private CallStack() {}

  /**
   * Returns the <em>full</em> call stack of the given rule, including both {@link Rule#getLocation}
   * and {@link Rule#getInteriorCallStack}.
   */
  static Node getFullCallStack(Rule rule) {
    return new Node(StarlarkThread.TOP_LEVEL, rule.getLocation(), rule.getInteriorCallStack());
  }

  /** Compact representation of a call stack entry. */
  @AutoCodec
  static final class Node {
    /** Function name. */
    private final String name;
    /** File name. */
    private final String file;

    private final int line;
    private final int col;
    @Nullable private final Node next;

    @AutoCodec.Instantiator
    @VisibleForSerialization
    static Node createForDeserialization(
        String name, String file, int line, int col, @Nullable Node next) {
      // Use common canonicalizer based on assertion that most strings (function names, locations)
      // were already shared across packages to some degree.
      return new Node(name.intern(), file.intern(), line, col, next);
    }

    private Node(String name, Location location, @Nullable Node next) {
      this(name, location.file(), location.line(), location.column(), next);
    }

    private Node(String name, String file, int line, int col, @Nullable Node next) {
      this.name = name;
      this.file = file;
      this.line = line;
      this.col = col;
      this.next = next;
    }

    Location toLocation() {
      return Location.fromFileLineColumn(file, line, col);
    }

    StarlarkThread.CallStackEntry toCallStackEntry() {
      return StarlarkThread.callStackEntry(name, toLocation());
    }

    String functionName() {
      return name;
    }

    @Nullable
    Node next() {
      return next;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Node)) {
        return false;
      }
      Node node = (Node) o;
      return line == node.line
          && col == node.col
          && name.equals(node.name)
          && file.equals(node.file)
          && Objects.equals(next, node.next);
    }

    @Override
    public int hashCode() {
      int result = HashCodes.hashObjects(name, file, next);
      result = result * 31 + Integer.hashCode(line);
      result = result * 31 + Integer.hashCode(col);
      return result;
    }
  }

  private static final Interner<Node> nodeInterner = BlazeInterners.newWeakInterner();

  /**
   * Returns a compact representation of the given call stack, optionally ignoring the outermost
   * frame.
   *
   * @param start index of frame at which to start; in other words, skip this many outermost frames.
   *     This is useful for skipping the outermost frame in BUILD file thread stacks, since the
   *     BUILD file location is already stored in {@link Rule#getLocation} and {@link
   *     MacroInstance#getBuildFileLocation}.
   * @return {@code null} for call stacks with fewer than two frames.
   */
  @Nullable
  static Node compact(List<StarlarkThread.CallStackEntry> stack, int start) {
    Node node = null;
    for (int i = stack.size() - 1; i >= start; i--) {
      StarlarkThread.CallStackEntry entry = stack.get(i);
      node = nodeInterner.intern(new Node(entry.name, entry.location, node));
    }
    return node;
  }

  /**
   * Returns a concatenation of two compact call stacks.
   *
   * <p>The result will contain {@code inner} stack appended unmodified to a new copy of the {@code
   * outer} stack.
   *
   * @return {@code null} if both of the inputs are {@code null} - in other words, if both of the
   *     inputs are empty stacks.
   */
  @Nullable
  static Node concatenate(@Nullable Node outer, @Nullable Node inner) {
    Deque<Node> outerReversed = new ArrayDeque<>();
    while (outer != null) {
      outerReversed.addFirst(outer);
      outer = outer.next();
    }
    Node node = inner;
    for (Node origOuterNode : outerReversed) {
      node =
          nodeInterner.intern(
              new Node(
                  origOuterNode.name,
                  origOuterNode.file,
                  origOuterNode.line,
                  origOuterNode.col,
                  node));
    }
    return node;
  }
}
