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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.Objects;

/**
 * A {@link NodeKey} is effectively a pair (type, name) that identifies a Skyframe node.
 */
public final class NodeKey implements Serializable {
  private final NodeType nodeType;

  /**
   * The name of the node.
   *
   * <p>This is deliberately an untyped Object so that we can use arbitrary value types (e.g.,
   * Labels, PathFragments, BuildConfigurations, etc.) as node names without incurring serialization
   * costs in the in-memory implementation of the graph.
   */
  private final Object nodeName;

  public NodeKey(NodeType nodeType, Object nodeName) {
    this.nodeType = Preconditions.checkNotNull(nodeType);
    this.nodeName = Preconditions.checkNotNull(nodeName);
  }

  public NodeType getNodeType() {
    return nodeType;
  }

  public Object getNodeName() {
    return nodeName;
  }

  @Override
  public String toString() {
    return nodeType + ":" + nodeName;
  }

  @Override
  public int hashCode() {
    return Objects.hash(nodeType, nodeName);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    NodeKey other = (NodeKey) obj;
    return nodeName.equals(other.nodeName) && nodeType.equals(other.nodeType);
  }

  public static final Function<NodeKey, Object> NODE_NAME = new Function<NodeKey, Object>() {
    @Override
    public Object apply(NodeKey input) {
      return input.getNodeName();
    }
  };
}
