// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Represents a collection of {@link FileOpNode}s, allowing for nested structures to represent
 * complex file dependencies.
 *
 * <p>This class serves as a container for multiple {@link FileOpNode} instances, enabling the
 * representation of file operation dependencies in a hierarchical manner. It differentiates between
 * analysis dependencies (for example, BUILD and .bzl files) and "source" dependencies, used during
 * execution (for example, .cpp, .h or .java files). It keeps them together to optimize storage.
 *
 * <p><b>Source vs. Analysis Dependencies:</b>
 *
 * <ul>
 *   <li><b>Analysis:</b> During the analysis phase, source files are declared, but configured
 *       targets (which define actions) do not depend on the <i>contents</i> of these source files,
 *       for example, .cpp, .h or .java files.
 *   <li><b>Execution:</b> The execution phase creates actual dependencies on the contents of source
 *       files as actions are run.
 * </ul>
 *
 * <p><b>Why combine them?</b> <br>
 * Logically, source and analysis dependencies could be tracked separately with different {@link
 * FileOpNode}s. However, this would duplicate the dependency graph structure in persistent storage,
 * which is expensive. This class keeps them together, trading off a bit of complexity for reduced
 * storage overhead. The structure is written only once, and the interpretation of dependencies must
 * be handled by the client.
 *
 * <p><b>Subclasses:</b>
 *
 * <ul>
 *   <li>{@link NestedFileOpNodes}: Represents a set of {@link FileOpNode}s without any immediate
 *       source file dependencies.
 *   <li>{@link NestedFileOpNodesWithSource}: Represents a set of {@link FileOpNode}s along with an
 *       immediate source file dependency ({@link FileKey}s).
 * </ul>
 */
public abstract sealed class AbstractNestedFileOpNodes implements FileOpNodeOrFuture.FileOpNode
    permits AbstractNestedFileOpNodes.NestedFileOpNodes,
        AbstractNestedFileOpNodes.NestedFileOpNodesWithSource {
  private final FileOpNode[] analysisDependencies;

  /**
   * Opaque storage for use by serialization.
   *
   * <p>{@link FileOpNode}, {@link FileKey} and {@link DirectoryListingKey} are mutually dependent
   * via {@link FileOpNode}. This type is opaque to avoid forcing {@link FileKey} and {@link
   * DirectoryListingKey} to depend on serialization implementation code.
   *
   * <p>The serialization implementation initializes this field with double-checked locking so it is
   * marked volatile.
   */
  private volatile Object serializationScratch;

  /**
   * Effectively, a factory method for {@link NestedFileOpNodes}, but formally a factory method for
   * {@link FileOpNodeOrEmpty}.
   *
   * <p>Returns {@link EMPTY_FILE_OP_NODE} if {@code analysisDependencies} is empty. When {@code
   * analysisDependencies} contains only one node, returns the node directly instead of wrapping it.
   * Otherwise, returns a {@link NestedFileOpNodes} instance wrapping {@code analysisDependencies}.
   */
  public static FileOpNodeOrEmpty from(Collection<FileOpNode> analysisDependencies) {
    if (analysisDependencies.isEmpty()) {
      return EMPTY_FILE_OP_NODE;
    }
    if (analysisDependencies.size() == 1) {
      return analysisDependencies.iterator().next();
    }
    return new NestedFileOpNodes(analysisDependencies.toArray(FileOpNode[]::new));
  }

  /**
   * Creates {@link NestedFileOpNodesWithSource} with reductions similar to {@link
   * #from(Collection<FileOpNode>)}.
   */
  public static FileOpNodeOrEmpty from(
      Collection<FileOpNode> analysisDependencies, @Nullable FileKey source) {
    if (source == null) {
      return from(analysisDependencies);
    }
    // It's unclear if `analysisDependencies` can ever be empty here in practice, but it's
    // permitted. It should be rare enough that defining a special type for it isn't worth it.
    return new NestedFileOpNodesWithSource(analysisDependencies.toArray(FileOpNode[]::new), source);
  }

  private AbstractNestedFileOpNodes(FileOpNode[] analysisDependencies) {
    this.analysisDependencies = analysisDependencies;
  }

  public int analysisDependenciesCount() {
    return analysisDependencies.length;
  }

  public FileOpNode getAnalysisDependency(int index) {
    return analysisDependencies[index];
  }

  /**
   * The number of walked parents of this node.
   *
   * <p>Tallied by the single-threaded leaf ID walk in {@code FileOpNodeMemoizingLookup}, which
   * finishes before serialization reads it, so a plain {@code int} suffices. The serializer seeds
   * this node's bitmap reference count with it. Each parent releases its reference once it has
   * merged this node's bitmap, so the bitmap is freed when the last parent is done. Root nodes,
   * which no walked node depends on, stay at 0.
   *
   * <p>{@code FileOpNodeCollector} gathers each parent's children in a set, so a parent lists a
   * child at most once. The count would still balance if one didn't, since the walk increments and
   * the serializer releases once per listing.
   */
  private int inDegree;

  @Nullable
  public Object getSerializationScratch() {
    return serializationScratch;
  }

  public void setSerializationScratch(Object value) {
    this.serializationScratch = value;
  }

  public void incrementInDegree() {
    inDegree++;
  }

  public int getInDegree() {
    return inDegree;
  }

  @VisibleForTesting
  public void addInDegreesForTesting(int delta) {
    inDegree += delta;
  }

  /**
   * This node's longest-path depth, or a traversal sentinel while the walk is still running.
   *
   * <p>A node that no walked node depends on has depth {@link #TOP_DEPTH}, and every other node is
   * one deeper than its deepest parent. That is, the depth is one more than the length of the
   * longest path to the node from a top node, not the shortest. Every edge therefore points to a
   * strictly greater depth, so visiting nodes deepest first places each node after all its
   * dependencies.
   *
   * <p>Doubles as the traversal's visited marker. A separate visited set over the whole node
   * population would cost tens of megabytes, which is the same order as the bitmap residency the
   * traversal exists to reduce.
   *
   * <p>Not volatile: written only by the single-threaded walk in {@link
   * com.google.devtools.build.lib.skyframe.serialization.analysis.FileOpNodeMemoizingLookup}, which
   * happens-before the serialization work that reads it.
   */
  private int longestPathDepth = UNDISCOVERED;

  /** Sentinel for a node the traversal has not reached yet. Matches the default field value. */
  private static final int UNDISCOVERED = 0;

  /** The depth of a node that no walked node depends on. */
  public static final int TOP_DEPTH = 1;

  public int getLongestPathDepth() {
    return longestPathDepth;
  }

  /**
   * Marks this node visited, provisionally at the top depth.
   *
   * @return false if the node was already visited
   */
  public boolean markVisited() {
    if (longestPathDepth != UNDISCOVERED) {
      return false;
    }
    longestPathDepth = TOP_DEPTH;
    return true;
  }

  /**
   * Places this node at least one deeper than a parent at {@code parentDepth}.
   *
   * <p>The node must already have been visited.
   */
  public void placeBelow(int parentDepth) {
    if (parentDepth >= longestPathDepth) {
      longestPathDepth = parentDepth + 1;
    }
  }

  /** A set of {@link FileOpNode}s with no immediate source dependencies. */
  public static final class NestedFileOpNodes extends AbstractNestedFileOpNodes {
    private NestedFileOpNodes(FileOpNode[] analysisDependencies) {
      super(analysisDependencies);
      checkArgument(analysisDependencies.length > 0);
    }
  }

  /** A set of analysis dependencies and source file dependency. */
  public static final class NestedFileOpNodesWithSource extends AbstractNestedFileOpNodes {
    private final FileKey source;

    private NestedFileOpNodesWithSource(FileOpNode[] nodes, FileKey source) {
      super(nodes);
      this.source = source;
    }

    public FileKey source() {
      return source;
    }
  }
}
