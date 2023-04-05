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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
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
 * Compact representation of a Starlark call stack for rule instantiations.
 *
 * <p>Implementation is optimized for minimizing memory overhead by sharing {@link Node} instances
 * when two call stacks have a common tail. For example, two different BUILD files that call into
 * the same macro can share {@link Node} instances for every frame except the top-level call.
 *
 * <p>The sharing rate of interior nodes is expected to be high, so nodes are implemented as a
 * linked list to eliminate array cost.
 */
public final class CallStack {

  /** Null instance, for use in testing or where content doesn't actually matter. */
  public static final CallStack EMPTY = new CallStack(/* size= */ 0, /* head= */ null);

  /** Number of frames in this stack. */
  private final int size;
  /** Top-level call in the call stack. */
  @Nullable final Node head;

  private CallStack(int size, @Nullable Node head) {
    this.size = size;
    this.head = head;
  }

  /** Returns the call stack as a list of frames, outermost call first. */
  public ImmutableList<StarlarkThread.CallStackEntry> toList() {
    StarlarkThread.CallStackEntry[] array = new StarlarkThread.CallStackEntry[size];
    int i = 0;
    for (Node n = head; n != null; n = n.next) {
      array[i++] = n.toCallStackEntry();
    }
    return ImmutableList.copyOf(array);
  }

  /** Returns a single frame, like {@code toList().get(i)} but more efficient. */
  StarlarkThread.CallStackEntry getFrame(int i) {
    for (Node n = head; n != null; n = n.next, i--) {
      if (i == 0) {
        return n.toCallStackEntry();
      }
    }
    throw new IndexOutOfBoundsException(); // !(0 <= i < size)
  }

  /** Returns the number of frames in the call stack. */
  public int size() {
    return size;
  }

  /** Compact representation of a call stack entry. */
  static final class Node {
    /** Function name. */
    private final String name;
    /** File name. */
    private final String file;

    private final int line;
    private final int col;
    @Nullable private final Node next;

    private Node(String name, String file, int line, int col, @Nullable Node next) {
      this.name = name;
      this.file = file;
      this.line = line;
      this.col = col;
      this.next = next;
    }

    private Location toLocation() {
      return Location.fromFileLineColumn(file, line, col);
    }

    private StarlarkThread.CallStackEntry toCallStackEntry() {
      return new StarlarkThread.CallStackEntry(name, toLocation());
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

  static CallStack createFrom(List<StarlarkThread.CallStackEntry> stack) {
    Node child = null;
    int n = stack.size();
    for (int i = n - 1; i >= 0; i--) {
      StarlarkThread.CallStackEntry entry = stack.get(i);

      String name = entry.name;
      String file = entry.location.file();
      int line = entry.location.line();
      int column = entry.location.column();

      child = new Node(name, file, line, column, child);
      if (i != 0) {
        // We don't intern top node, because it hardly ever repeats.
        child = nodeInterner.intern(child);
      }
    }

    // Use the same node for all empty stacks, to avoid allocations.
    return child != null ? new CallStack(n, child) : EMPTY;
  }

  /**
   * Efficient serializer for {@link CallStack}s. Before callstacks are serialized in a package
   * {@link #prepareCallStack(CallStack)} method must be called on them (to prepare a table of
   * strings).
   */
  static class Serializer {
    private static final int NULL_NODE_ID = 0;

    private final IdentityHashMap<Node, Integer> nodeTable = new IdentityHashMap<>();
    private final List<String> stringTable = new ArrayList<>();
    private boolean stringTableSerialized = false;

    private final Map<String, Integer> stringTableIndex = new HashMap<>();

    private int indexOf(String s) {
      int i = stringTableIndex.size();
      Integer prev = stringTableIndex.putIfAbsent(s, i);
      if (prev != null) {
        i = prev;
      } else {
        checkArgument(
            !stringTableSerialized, "Can only serialize CallStacks that were prepared before.");
        stringTable.add(s);
      }
      return i;
    }

    Serializer() {
      nodeTable.put(null, NULL_NODE_ID);
    }

    void serializeCallStack(CallStack callStack, CodedOutputStream codedOut) throws IOException {
      if (!stringTableSerialized) {
        codedOut.writeInt32NoTag(stringTable.size());
        for (String string : stringTable) {
          codedOut.writeStringNoTag(string);
        }
        stringTableSerialized = true;
      }

      codedOut.writeInt32NoTag(callStack.size);
      emitNode(callStack.head, codedOut);
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

    void prepareCallStack(CallStack callStack) {
      for (Node n = callStack.head; n != null; n = n.next) {
        indexOf(n.name);
        indexOf(n.file);
      }
    }
  }

  /** Deserializes {@link CallStack}s as serialized by a {@link Serializer}. */
  static class Deserializer {
    private static final Node DUMMY_NODE = new Node("", "", -1, -1, null);

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
      return new CallStack(size, readNode(codedIn));
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
