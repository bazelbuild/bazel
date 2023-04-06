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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
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

  /** Compact representation of a call stack entry. */
  static final class Node {
    /** Function name. */
    private final String name;
    /** File name. */
    private final String file;

    private final int line;
    private final int col;
    @Nullable private final Node next;

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
   * Returns a compact representation of the <em>interior</em> of the given call stack.
   *
   * <p>The outermost frame of the call stack is not reflected in the compact representation because
   * it is already stored as {@link Rule#getLocation}.
   *
   * <p>Returns {@code null} for call stacks with fewer than two frames.
   */
  @Nullable
  static Node compactInterior(List<StarlarkThread.CallStackEntry> stack) {
    Node node = null;
    for (int i = stack.size() - 1; i > 0; i--) {
      StarlarkThread.CallStackEntry entry = stack.get(i);
      node = nodeInterner.intern(new Node(entry.name, entry.location, node));
    }
    return node;
  }

  /**
   * Efficient serializer for {@link Node}s. Before callstacks are serialized in a package {@link
   * #prepareCallStack} method must be called on them (to prepare a table of strings).
   */
  static final class Serializer {
    private static final int NULL_NODE_ID = 0;

    private final IdentityHashMap<Node, Integer> nodeTable = new IdentityHashMap<>();
    private final List<String> stringTable = new ArrayList<>();
    private boolean stringTableSerialized = false;

    private final Map<String, Integer> stringTableIndex = new HashMap<>();

    Serializer() {
      nodeTable.put(null, NULL_NODE_ID);
      indexString(StarlarkThread.TOP_LEVEL);
    }

    private void indexString(String s) {
      checkState(!stringTableSerialized);
      int i = stringTableIndex.size();
      if (stringTableIndex.putIfAbsent(s, i) == null) {
        stringTable.add(s);
      }
    }

    private int indexOf(String s) {
      return checkNotNull(stringTableIndex.get(s), s);
    }

    /**
     * Serializes the <em>full</em> call stack of the given rule, including both {@link
     * Rule#getLocation} and {@link Rule#getInteriorCallStack}.
     */
    void serializeCallStack(Rule rule, CodedOutputStream codedOut) throws IOException {
      if (!stringTableSerialized) {
        codedOut.writeInt32NoTag(stringTable.size());
        for (String string : stringTable) {
          codedOut.writeStringNoTag(string);
        }
        stringTableSerialized = true;
      }

      emitNode(
          new Node(StarlarkThread.TOP_LEVEL, rule.getLocation(), rule.getInteriorCallStack()),
          codedOut);
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
      codedOut.writeInt32NoTag(indexOf(node.name));
      codedOut.writeInt32NoTag(indexOf(node.file));
      codedOut.writeInt32NoTag(node.line);
      codedOut.writeInt32NoTag(node.col);
      emitNode(node.next, codedOut);
    }

    void prepareCallStack(Rule rule) {
      indexString(rule.getLocation().file());
      for (Node n = rule.getInteriorCallStack(); n != null; n = n.next) {
        indexString(n.name);
        indexString(n.file);
      }
    }
  }

  /** Deserializes call stacks as serialized by a {@link Serializer}. */
  static final class Deserializer {
    private static final Node DUMMY_NODE = new Node("", "", -1, -1, null);

    private final List<Node> nodeTable = new ArrayList<>();
    @Nullable private List<String> stringTable;

    Deserializer() {
      // By convention index 0 = null.
      nodeTable.add(null);
    }

    /**
     * Deserializes a <em>full</em> call stack.
     *
     * <p>The returned {@link Node} represents {@link Rule#getLocation}. Calling {@link Node#next()}
     * on the returned {@link Node} yields {@link Rule#getInteriorCallStack}.
     */
    Node deserializeFullCallStack(CodedInputStream codedIn) throws IOException {
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

      return readNode(codedIn);
    }

    @Nullable
    @SuppressWarnings("ReferenceEquality")
    private Node readNode(CodedInputStream codedIn) throws IOException {
      int index = codedIn.readInt32();
      if (index < nodeTable.size()) {
        Node result = nodeTable.get(index);
        checkState(result != DUMMY_NODE, "Loop detected at index %s", index);
        return result;
      }

      checkState(
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
      Node child = readNode(codedIn);

      Node result = new Node(stringTable.get(name), stringTable.get(file), line, col, child);
      nodeTable.set(index, result);
      return result;
    }
  }
}
