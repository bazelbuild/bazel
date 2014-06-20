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

import com.google.common.base.Predicate;

import java.io.Serializable;
import java.util.Set;

/**
 * The type of a node.
 */
public final class NodeType implements Serializable {
  public static NodeType computed(String name) {
    return new NodeType(name, true);
  }

  private final String name;
  private final boolean isComputed;

  public NodeType(String name, boolean isComputed) {
    this.name = name;
    this.isComputed = isComputed;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof NodeType)) {
      return false;
    }
    NodeType other = (NodeType) obj;
    return name.equals(other.name);
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }

  /**
   * Returns whether the nodes of this type are computed. The computation of a computed node must
   * be deterministic and may only access requested dependencies.
   */
  public boolean isComputed() {
    return isComputed;
  }

  /**
   * A predicate that returns true for {@link NodeKey}s that have the given {@link NodeType}.
   */
  public static Predicate<NodeKey> nodeTypeIs(final NodeType nodeType) {
    return new Predicate<NodeKey>() {
      @Override
      public boolean apply(NodeKey nodeKey) {
        return nodeKey.getNodeType() == nodeType;
      }
    };
  }

  /**
   * A predicate that returns true for {@link NodeKey}s that have the given {@link NodeType}.
   */
  public static Predicate<NodeKey> nodeTypeIsIn(final Set<NodeType> nodeTypes) {
    return new Predicate<NodeKey>() {
      @Override
      public boolean apply(NodeKey nodeKey) {
        return nodeTypes.contains(nodeKey.getNodeType());
      }
    };
  }
}
