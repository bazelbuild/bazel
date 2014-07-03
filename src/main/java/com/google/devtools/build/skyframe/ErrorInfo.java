// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Information about why a {@link Node} failed to evaluate successfully.
 *
 * <p>This is intended only for use in alternative {@code AutoUpdatingGraph} implementations.
 */
public class ErrorInfo {
  /**
   * The set of descendants of this node that failed to build
   */
  private final NestedSet<NodeKey> rootCauses;

  /**
   * An exception thrown upon a node's failure to build. The exception is used for reporting, and
   * thus may ultimately be rethrown by the caller. As well, during a --nokeep_going evaluation, if
   * an error node is encountered from an earlier --keep_going build, the exception to be thrown is
   * taken from here.
   */
  @Nullable private final Throwable exception;

  private final Iterable<CycleInfo> cycles;

  private final boolean isTransient;

  public ErrorInfo(NodeBuilderException builderException) {
    this.rootCauses = NestedSetBuilder.create(Order.STABLE_ORDER,
        Preconditions.checkNotNull(builderException.getRootCauseNodeKey(), builderException));
    this.exception = Preconditions.checkNotNull(builderException.getCause(), builderException);
    this.cycles = ImmutableList.of();
    this.isTransient = builderException.isTransient();
  }

  ErrorInfo(CycleInfo cycleInfo) {
    this.rootCauses = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    this.exception = null;
    this.cycles = ImmutableList.of(cycleInfo);
    this.isTransient = false;
  }

  public ErrorInfo(NodeKey currentNode, Collection<ErrorInfo> childErrors) {
    Preconditions.checkNotNull(currentNode);
    Preconditions.checkState(!childErrors.isEmpty(),
        "Error node %s with no exception must depend on another error node", currentNode);
    NestedSetBuilder<NodeKey> builder = NestedSetBuilder.stableOrder();
    ImmutableList.Builder<CycleInfo> cycleBuilder = ImmutableList.builder();
    Throwable firstException = null;
    boolean isTransient = false;
    for (ErrorInfo child : childErrors) {
      if (firstException == null) {
        firstException = child.getException();
      }
      builder.addTransitive(child.rootCauses);
      cycleBuilder.addAll(CycleInfo.prepareCycles(currentNode, child.cycles));
      isTransient |= child.isTransient();
    }
    this.rootCauses = builder.build();
    this.exception = firstException;
    this.cycles = cycleBuilder.build();
    this.isTransient = isTransient;
  }

  @Override
  public String toString() {
    return String.format("<ErrorInfo exception=%s rootCauses=%s cycles=%s>",
        rootCauses, exception, cycles);
  }

  /**
   * The root causes of a node that failed to build are its descendant nodes that failed to build.
   * If a node's descendants all built successfully, but it failed to, its root cause will be
   * itself. If a node depends on a cycle, but has no other errors, this method will return
   * the empty set.
   */
  public Iterable<NodeKey> getRootCauses() {
    return rootCauses;
  }

  /**
   * The exception thrown when building a node. May be null if node's only error is depending
   * on a cycle.
   */
  @Nullable public Throwable getException() {
    return exception;
  }

  /**
   * Any cycles found when building this node.
   *
   * <p>If there are a large number of cycles, only a limited number are returned here.
   *
   * <p>If this node has a child through which there are multiple paths to the same cycle, only one
   * path is returned here. However, if there are multiple paths to the same cycle, each of which
   * goes through a different child, each of them is returned here.
   */
  public Iterable<CycleInfo> getCycleInfo() {
    return cycles;
  }

  /**
   * Returns true iff the error is transient, i.e. if retrying the same computation could lead to a
   * different result.
   */
  public boolean isTransient() {
    return isTransient;
  }
}
