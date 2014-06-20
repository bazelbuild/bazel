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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * Encapsulation of data stored by {@link NodeEntry} when the node has finished building.
 */
abstract class NodeWithMetadata implements Node {
  protected final Node node;

  private static final NestedSet<TaggedEvents> NO_EVENTS =
      NestedSetBuilder.<TaggedEvents>emptySet(Order.STABLE_ORDER);

  public NodeWithMetadata(Node node) {
    this.node = node;
  }

  /** Build a node entry value that has an error (and no node value). */
  static NodeWithMetadata error(ErrorInfo errorInfo, NestedSet<TaggedEvents> transitiveEvents) {
    return new ErrorInfoNode(errorInfo, null, transitiveEvents);
  }

  /**
   * Build a node entry value that has a node value, and possibly an error (constructed from its
   * children's errors).
   */
  static Node normal(@Nullable Node node, @Nullable ErrorInfo errorInfo,
      NestedSet<TaggedEvents> transitiveEvents) {
    Preconditions.checkState(node != null || errorInfo != null,
        "Value and error cannot both be null");
    if (errorInfo == null) {
      return transitiveEvents.isEmpty()
          ? node
          : new NodeWithEvents(node, transitiveEvents);
    }
    return new ErrorInfoNode(errorInfo, node, transitiveEvents);
  }


  @Nullable Node getNode() {
    return node;
  }

  @Nullable
  abstract ErrorInfo getErrorInfo();

  abstract NestedSet<TaggedEvents> getTransitiveEvents();

  static final class NodeWithEvents extends NodeWithMetadata {

    private final NestedSet<TaggedEvents> transitiveEvents;

    NodeWithEvents(Node node, NestedSet<TaggedEvents> transitiveEvents) {
      super(Preconditions.checkNotNull(node));
      this.transitiveEvents = Preconditions.checkNotNull(transitiveEvents);
    }

    @Nullable
    @Override
    ErrorInfo getErrorInfo() { return null; }

    @Override
    NestedSet<TaggedEvents> getTransitiveEvents() { return transitiveEvents; }

    /**
     * We override equals so that if the same value is written to a {@link NodeEntry} twice, it can
     * verify that the two values are equal, and avoid incrementing its version.
     */
    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }

      NodeWithEvents that = (NodeWithEvents) o;

      // Shallow equals is a middle ground between using default equals, which might miss
      // nested sets with the same elements, and deep equality checking, which would be expensive.
      // All three choices are sound, since shallow equals and default equals are more
      // conservative than deep equals. Using shallow equals means that we may unnecessarily
      // consider some nodes unequal that are actually equal, but this is still a net win over
      // deep equals.
      return node.equals(that.node) && transitiveEvents.shallowEquals(that.transitiveEvents);
    }

    @Override
    public int hashCode() {
      return 31 * node.hashCode() + transitiveEvents.hashCode();
    }

    @Override
    public String toString() { return node.toString(); }
  }

  static final class ErrorInfoNode extends NodeWithMetadata {

    private final ErrorInfo errorInfo;
    private final NestedSet<TaggedEvents> transitiveEvents;

    ErrorInfoNode(ErrorInfo errorInfo, @Nullable Node node,
        NestedSet<TaggedEvents> transitiveEvents) {
      super(node);
      this.errorInfo = Preconditions.checkNotNull(errorInfo);
      this.transitiveEvents = Preconditions.checkNotNull(transitiveEvents);
    }

    @Nullable
    @Override
    ErrorInfo getErrorInfo() { return errorInfo; }

    @Override
    NestedSet<TaggedEvents> getTransitiveEvents() { return transitiveEvents; }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }

      ErrorInfoNode that = (ErrorInfoNode) o;

      // Shallow equals is a middle ground between using default equals, which might miss
      // nested sets with the same elements, and deep equality checking, which would be expensive.
      // All three choices are sound, since shallow equals and default equals are more
      // conservative than deep equals. Using shallow equals means that we may unnecessarily
      // consider some nodes unequal that are actually equal, but this is still a net win over
      // deep equals.
      return Objects.equals(this.node, that.node) && Objects.equals(this.errorInfo, that.errorInfo)
          && transitiveEvents.shallowEquals(that.transitiveEvents);
    }

    @Override
    public int hashCode() {
      return 31 * Objects.hash(node, errorInfo) + transitiveEvents.shallowHashCode();
    }

    @Override
    public String toString() {
      StringBuilder result = new StringBuilder();
      if (node != null) {
        result.append("Value: ").append(node);
      }
      if (errorInfo != null) {
        if (result.length() > 0) {
          result.append("; ");
        }
        result.append("Error: ").append(errorInfo);
      }
      return result.toString();
    }
  }

  static Node justNode(Node node) {
    if (node instanceof NodeWithMetadata) {
      return ((NodeWithMetadata) node).getNode();
    }
    return node;
  }

  static NodeWithMetadata wrapWithMetadata(Node node) {
    if (node instanceof NodeWithMetadata) {
      return (NodeWithMetadata) node;
    }
    return new NodeWithEvents(node, NO_EVENTS);
  }

  @Nullable
  static ErrorInfo getMaybeErrorInfo(Node node) {
    if (node.getClass() == ErrorInfoNode.class) {
      return ((NodeWithMetadata) node).getErrorInfo();
    }
    return null;

  }
}