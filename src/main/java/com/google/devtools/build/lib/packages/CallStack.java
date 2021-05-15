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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Optimized representation of a Starlark call stack.
 *
 * <p>Implementation is optimized for minimizing the overhead at a package level. All {@link
 * CallStack}s created from the same {@link Factory} share internal state, so all {@link CallStack}s
 * in a package are expected to be created from the same {@link Factory}.
 */
public final class CallStack {

  /** Null instance, for use in testing or where contents doesn't actually matter. */
  public static final CallStack EMPTY = new CallStack(ImmutableList.of(), 0, null);

  /** String table, shared with all instances created from the same {@link Factory}. */
  private final List<String> strings;
  /** Number of frames in this stack. */
  private final int size;
  /** Top (innermost call) of the call stack. */
  @Nullable private final Node node;

  private CallStack(List<String> strings, int size, @Nullable Node node) {
    this.strings = strings;
    this.size = size;
    this.node = node;
  }

  /** Returns the call stack as a list of frames, outermost call first. */
  public ImmutableList<StarlarkThread.CallStackEntry> toList() {
    StarlarkThread.CallStackEntry[] array = new StarlarkThread.CallStackEntry[size];
    int i = size;
    for (Node n = node; n != null; n = n.parent) {
      array[--i] = nodeFrame(n);
    }
    return ImmutableList.copyOf(array);
  }

  /** Returns a single frame, like {@code toList().get(i)} but more efficient. */
  public StarlarkThread.CallStackEntry getFrame(int i) {
    for (Node n = node; n != null; n = n.parent) {
      if (++i == size) {
        return nodeFrame(n);
      }
    }
    throw new IndexOutOfBoundsException(); // !(0 <= i < size)
  }

  /** Returns the number of frames in the call stack. */
  public int size() {
    return size;
  }

  private StarlarkThread.CallStackEntry nodeFrame(Node n) {
    String file = strings.get(n.file);
    String name = strings.get(n.name);
    Location loc = Location.fromFileLineColumn(file, n.line, n.col);
    return new StarlarkThread.CallStackEntry(name, loc);
  }

  /** Compact representation of a call stack entry. */
  private static class Node {
    /** Index of function name. */
    private final int name;
    /** Index of file name. */
    private final int file;

    private final int line;
    private final int col;
    @Nullable private final Node parent;

    Node(int name, int file, int line, int col, Node parent) {
      this.name = name;
      this.file = file;
      this.line = line;
      this.col = col;
      this.parent = parent;
    }
  }

  /**
   * Preferred instantiation method. All {@link CallStack} instances produced from a {@link Factory}
   * will share some amount of internal state.
   *
   * <p>All {@link CallStack}s in a package should be created from the same {@link Factory}
   * instance, and there should be exactly one {@link Factory} instance per package.
   */
  static final class Factory {

    private final Map<String, Integer> stringTableIndex = new HashMap<>();
    private final List<String> stringTable = new ArrayList<>();
    /** Unmodifiable view of the string table to be shared with instances. */
    private final List<String> unmodifiableStringTable = Collections.unmodifiableList(stringTable);

    /**
     * Previously encountered call stack. This is an optimization to take advantage of the
     * observation that sequentially created instances are likely to overlapping call-stacks due to
     * coming from sequentially created rules.
     */
    private Node[] nodes = new Node[10];
    /** Depth of previously encountered call stack. */
    private int depth = 0;

    /**
     * Returns a {@link CallStack}.
     *
     * <p>Space efficiency depends on the similarity of the list elements in successive calls.
     * Reversing the stack before calling this function destroys the optimization, for instance.
     */
    CallStack createFrom(List<StarlarkThread.CallStackEntry> stack) {
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
            && parent == nodes[i].parent
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
      return parent != null ? new CallStack(unmodifiableStringTable, n, parent) : EMPTY;
    }

    private int indexOf(String s) {
      int i = stringTableIndex.size();
      Integer prev = stringTableIndex.putIfAbsent(s, i);
      if (prev != null) {
        i = prev;
      } else {
        stringTable.add(s);
      }
      return i;
    }
  }

  /**
   * Efficient serializer for {@link CallStack}s. All {@link CallStack}s instances passed to a
   * {@link Serializer} <b>MUST</b> have originated from the same {@link Factory} instance.
   */
  static class Serializer {
    private static final int NULL_NODE_ID = 0;

    private final IdentityHashMap<Node, Integer> nodeTable = new IdentityHashMap<>();
    @Nullable private List<String> stringTable;

    Serializer() {
      nodeTable.put(null, NULL_NODE_ID);
    }

    void serializeCallStack(CallStack callStack, CodedOutputStream codedOut) throws IOException {
      if (stringTable == null) {
        codedOut.writeInt32NoTag(callStack.strings.size());
        for (String string : callStack.strings) {
          codedOut.writeStringNoTag(string);
        }
        stringTable = callStack.strings;
      } else {
        Preconditions.checkArgument(
            stringTable == callStack.strings,
            "Can only serialize CallStacks that share a string table.");
      }

      codedOut.writeInt32NoTag(callStack.size);
      emitNode(callStack.node, codedOut);
    }

    private void emitNode(Node node, CodedOutputStream codedOut) throws IOException {
      Integer index = nodeTable.get(node);
      if (index != null) {
        codedOut.writeInt32NoTag(index);
        return;
      }

      if (node == null) {
        return;
      }

      int newIndex = nodeTable.size();
      codedOut.writeInt32NoTag(newIndex);
      nodeTable.put(node, newIndex);
      codedOut.writeInt32NoTag(node.name);
      codedOut.writeInt32NoTag(node.file);
      codedOut.writeInt32NoTag(node.line);
      codedOut.writeInt32NoTag(node.col);
      emitNode(node.parent, codedOut);
    }
  }

  /**
   * Deserializes {@link CallStack}s as serialized by a {@link Serializer}. Deserialized instances
   * are optimized as if they had been created from the same {@link Factory}.
   */
  static class Deserializer {
    private static final Node DUMMY_NODE = new Node(-1, -1, -1, -1, null);

    private final List<Node> nodeTable = new ArrayList<>();
    @Nullable private List<String> stringTable;

    Deserializer() {
      // By convention index 0 = null.
      nodeTable.add(null);
    }

    CallStack deserializeCallStack(CodedInputStream codedIn) throws IOException {
      if (stringTable == null) {
        int length = codedIn.readInt32();
        stringTable = new ArrayList<>(length);
        for (int i = 0; i < length; i++) {
          // Avoid having a new set of strings per deserialized string table. Use common
          // canonicalizer based on assertion that most strings (function names, locations) were
          // already some degree of shared across packages.
          stringTable.add(StringCanonicalizer.intern(codedIn.readString()));
        }
      }

      int size = codedIn.readInt32();
      return new CallStack(stringTable, size, readNode(codedIn));
    }

    @Nullable
    private Node readNode(CodedInputStream codedIn) throws IOException {
      int index = codedIn.readInt32();
      if (index < nodeTable.size()) {
        Node result = nodeTable.get(index);
        Preconditions.checkState(result != DUMMY_NODE, "Loop detected at index %s", index);
        return result;
      }

      Preconditions.checkState(
          index == nodeTable.size(),
          "Unexpected next value index - read %s, expected %s",
          index,
          nodeTable.size());

      // Add dummy node to grow the table and save our spot in the table until we're done.
      nodeTable.add(DUMMY_NODE);
      int name = codedIn.readInt32();
      int file = codedIn.readInt32();
      int line = codedIn.readInt32();
      int col = codedIn.readInt32();
      Node parent = readNode(codedIn);

      Node result = new Node(name, file, line, col, parent);
      nodeTable.set(index, result);
      return result;
    }
  }
}
