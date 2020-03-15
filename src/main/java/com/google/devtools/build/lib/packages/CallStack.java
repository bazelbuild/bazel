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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A CallStack is an opaque immutable stack of Starlark call frames, outermost call first. Its
 * representation is highly optimized for space. Call {@link #toArray} to access the frames.
 *
 * <p>A CallStack cannot be constructed directly, but must be created by calling the {@code of}
 * method of a Builder, which should be shared across all the rules of a package.
 */
public final class CallStack {

  // A naive implementation using a simple array of CallStackEntry was
  // found to increase retained heap size for a large query by about 6%.
  // We reduce this using two optimizations:
  //
  // 1) entry compression
  // A CallStackEntry has size 2 (header) + 1 String + 1 Location = 4 words.
  // A Location has size 2 (header) + 1 String + 1 (file/loc) = 4 words.
  // By representing the two strings as small integers---indices into a shared
  // table---we can represent the payload using 4 ints = 2 words.
  // We reconstitute new Locations and CallStackEntries on request.
  // Eliminating Java object overheads reduces the marginal
  // space for an element by about a half.
  //
  // 2) prefix sharing
  // We share common stack prefixes between successive entries.
  // The stacks passed to successive of() calls display locality because
  // within a package, many rules are created by a single macro or stack
  // of macros. The Builder records the previous stack and creates a tree node
  // for each prefix. When it sees a new stack, it finds the common prefix
  // for the current and previous stacks, and only creates new nodes for
  // the suffix of different entries.
  // Avoiding storage of redundant prefixes reduces overall space by about
  // two thirds.
  //
  // This class intentionally does not implement List
  // because internally it is a linked list,
  // so accidentally using List.get for sequential access
  // would result in quadratic behavior.

  public static final CallStack EMPTY = new CallStack(ImmutableList.of(), 0, null);

  private final List<String> strings; // builder's string table, shared
  private final int size; // depth of stack
  @Nullable private final Node node; // tree node representing this stack

  private CallStack(List<String> strings, int size, @Nullable Node node) {
    this.strings = strings;
    this.size = size;
    this.node = node;
  }

  /** Returns the call stack as a list of frames, outermost call first. */
  public ImmutableList<StarlarkThread.CallStackEntry> toList() {
    return ImmutableList.copyOf(toArray());
  }

  /** Returns the call stack as a new array, outermost call first. */
  public StarlarkThread.CallStackEntry[] toArray() {
    StarlarkThread.CallStackEntry[] array = new StarlarkThread.CallStackEntry[size];
    int i = size;
    for (Node n = node; n != null; n = n.parent) {
      String file = strings.get(n.file);
      String name = strings.get(n.name);
      Location loc = Location.fromFileLineColumn(file, n.line, n.col);
      array[--i] = new StarlarkThread.CallStackEntry(name, loc);
    }
    return array;
  }

  // A Node is a node in a linked tree representing a prefix of the call stack.
  // file and line are indices of strings in the builder's shared table.
  private static class Node {
    Node(int name, int file, int line, int col, Node parent) {
      this.name = name;
      this.file = file;
      this.line = line;
      this.col = col;
      this.parent = parent;
    }

    final int name;
    final int file;
    final int line;
    final int col;
    @Nullable final Node parent;
  }

  /** A Builder is a stateful indexer of call stacks. */
  static final class Builder {

    // string table: strings[index[s]] == s
    private final Map<String, Integer> index = new HashMap<>();
    private final List<String> strings = new ArrayList<>();

    // nodes[0:depth] is the previously encountered call stack.
    // We avoid ArrayList because we need efficient truncation.
    private Node[] nodes = new Node[10];
    private int depth = 0;

    /**
     * Returns a compact representation of the specified call stack. <br>
     * Space efficiency depends on the similarity of the list elements in successive calls.
     * Reversing the stack before calling this function destroys the optimization, for instance.
     */
    CallStack of(List<StarlarkThread.CallStackEntry> stack) {
      // We find and reuse the common ancestor node for the prefix common
      // to the current stack and the stack passed to the previous call,
      // then add child nodes for the different suffix, if any.

      // Loop invariant: parent == (i > 0 ? nodes[i-1] : null)
      Node parent = null;
      int n = stack.size();
      for (int i = 0; i < n; i++) {
        StarlarkThread.CallStackEntry entry = stack.get(i);

        int name = indexOf(entry.name);
        int file = indexOf(entry.location.file());
        int line = entry.location.line();
        int column = entry.location.column();
        if (i < depth
            && name == nodes[i].name
            && file == nodes[i].file
            && line == nodes[i].line
            && column == nodes[i].col) {
          parent = nodes[i];
          continue;
        }
        parent = new Node(name, file, line, column, parent);
        if (i == nodes.length) {
          nodes = Arrays.copyOf(nodes, nodes.length << 1); // grow by doubling
        }
        nodes[i] = parent;
      }
      this.depth = n; // truncate

      // Use the same node for all empty stacks, to avoid allocations.
      return parent != null ? new CallStack(strings, n, parent) : EMPTY;
    }

    private int indexOf(String s) {
      int i = index.size();
      Integer prev = index.putIfAbsent(s, i);
      if (prev != null) {
        i = prev;
      } else {
        strings.add(s);
      }
      return i;
    }
  }
}
