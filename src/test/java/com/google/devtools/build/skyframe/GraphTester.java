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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.util.Pair;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A helper class to create graphs and run skyframe tests over these graphs.
 *
 * <p>There are two types of nodes, computing nodes, which may not be set to a constant value,
 * and leaf nodes, which must be set to a constant value and may not have any dependencies.
 *
 * <p>Note that the node builder looks into the test nodes created here to determine how to
 * behave. However, skyframe will only re-evaluate the node and call the node builder if any of
 * its dependencies has changed. That means in order to change the set of dependencies of a node,
 * you need to also change one of its previous dependencies to force re-evaluation. Changing a
 * computing node does not mark it as modified.
 */
public class GraphTester {

  // TODO(bazel-team): Split this for computing and non-computing nodes?
  public static final NodeType NODE_TYPE = new NodeType("Type", false);

  private final Map<NodeKey, TestNodeBuilder> nodes = new HashMap<>();
  private final Set<NodeKey> modifiedNodes = new LinkedHashSet<>();

  public TestNodeBuilder getOrCreate(String name) {
    return getOrCreate(nodeKey(name));
  }

  public TestNodeBuilder getOrCreate(NodeKey key) {
    return getOrCreate(key, false);
  }

  public TestNodeBuilder getOrCreate(NodeKey key, boolean markAsModified) {
    TestNodeBuilder result = nodes.get(key);
    if (result == null) {
      result = new TestNodeBuilder();
      nodes.put(key, result);
    } else if (markAsModified) {
      modifiedNodes.add(key);
    }
    return result;
  }

  public TestNodeBuilder set(String key, Node value) {
    return set(nodeKey(key), value);
  }

  public TestNodeBuilder set(NodeKey key, Node value) {
    return getOrCreate(key, true).setConstantValue(value);
  }

  public Collection<NodeKey> getModifiedNodes() {
    return modifiedNodes;
  }

  public NodeBuilder getNodeBuilder() {
    return new NodeBuilder() {
      @Override
      public Node build(NodeKey key, Environment env)
          throws NodeBuilderException, InterruptedException {
        TestNodeBuilder builder = nodes.get(key);
        if (builder.builder != null) {
          return builder.builder.build(key, env);
        }
        if (builder.warning != null) {
          env.getListener().warn(null, builder.warning);
        }
        if (builder.progress != null) {
          env.getListener().progress(null, builder.progress);
        }
        Map<NodeKey, Node> deps = new LinkedHashMap<>();
        boolean oneMissing = false;
        for (Pair<NodeKey, Node> dep : builder.deps) {
          Node value;
          if (dep.second == null) {
            value = env.getDep(dep.first);
          } else {
            try {
              value = env.getDepOrThrow(dep.first, SomeErrorException.class);
            } catch (SomeErrorException e) {
              value = dep.second;
            }
          }
          if (value == null) {
            oneMissing = true;
          } else {
            deps.put(dep.first, value);
          }
          Preconditions.checkState(oneMissing == env.depsMissing());
        }
        if (env.depsMissing()) {
          return null;
        }

        if (builder.hasError) {
          throw new GenericNodeBuilderException(key, new SomeErrorException(key.toString()));
        }
        if (builder.hasNonTransientError) {
          throw new GenericNodeBuilderException(key, new SomeErrorException(key.toString()), false);
        }

        if (builder.value != null) {
          return builder.value;
        }

        if (Thread.currentThread().isInterrupted()) {
          throw new InterruptedException(key.toString());
        }

        return builder.computer.compute(deps, env);
      }

      @Nullable
      @Override
      public String extractTag(NodeKey nodeKey) {
        return nodes.get(nodeKey).tag;
      }
    };
  }

  public static NodeKey nodeKey(String key) {
    return new NodeKey(NODE_TYPE, key);
  }

  public static class SomeErrorException extends Exception {
    public SomeErrorException(String msg) {
      super(msg);
    }
  }

  /**
   * A node in the testing graph that is constructed in the tester.
   */
  public class TestNodeBuilder {
    // TODO(bazel-team): We could use a multiset here to simulate multi-pass dependency discovery.
    private final Set<Pair<NodeKey, Node>> deps = new LinkedHashSet<>();
    private Node value;
    private NodeComputer computer;
    private NodeBuilder builder = null;

    private boolean hasError;
    private boolean hasNonTransientError;

    private String warning;
    private String progress;

    private String tag;

    public TestNodeBuilder addDependency(String name) {
      return addDependency(nodeKey(name));
    }

    public TestNodeBuilder addDependency(NodeKey key) {
      deps.add(Pair.<NodeKey, Node>of(key, null));
      return this;
    }

    public TestNodeBuilder removeDependency(String name) {
      return removeDependency(nodeKey(name));
    }

    public TestNodeBuilder removeDependency(NodeKey key) {
      deps.remove(Pair.<NodeKey, Node>of(key, null));
      return this;
    }

    public TestNodeBuilder addErrorDependency(String name, Node altValue) {
      return addErrorDependency(nodeKey(name), altValue);
    }

    public TestNodeBuilder addErrorDependency(NodeKey key, Node altValue) {
      deps.add(Pair.of(key, altValue));
      return this;
    }

    public TestNodeBuilder setConstantValue(Node value) {
      Preconditions.checkState(this.computer == null);
      this.value = value;
      return this;
    }

    public TestNodeBuilder setComputedValue(NodeComputer computer) {
      Preconditions.checkState(this.value == null);
      this.computer = computer;
      return this;
    }

    public TestNodeBuilder setBuilder(NodeBuilder builder) {
      Preconditions.checkState(this.value == null);
      Preconditions.checkState(this.computer == null);
      Preconditions.checkState(deps.isEmpty());
      Preconditions.checkState(!hasError);
      Preconditions.checkState(!hasNonTransientError);
      Preconditions.checkState(warning == null);
      Preconditions.checkState(progress == null);
      this.builder = builder;
      return this;
    }

    public TestNodeBuilder setHasError(boolean hasError) {
      this.hasError = hasError;
      return this;
    }

    public TestNodeBuilder setHasNonTransientError(boolean hasError) {
      // TODO(bazel-team): switch to an enum for hasError.
      this.hasNonTransientError = hasError;
      return this;
    }

    public TestNodeBuilder setWarning(String warning) {
      this.warning = warning;
      return this;
    }

    public TestNodeBuilder setProgress(String info) {
      this.progress = info;
      return this;
    }

    public TestNodeBuilder setTag(String tag) {
      this.tag = tag;
      return this;
    }

  }

  public static NodeKey[] toNodeKeys(String... names) {
    NodeKey[] result = new NodeKey[names.length];
    for (int i = 0; i < names.length; i++) {
      result[i] = new NodeKey(GraphTester.NODE_TYPE, names[i]);
    }
    return result;
  }

  public static NodeKey toNodeKey(String name) {
    return toNodeKeys(name)[0];
  }

  private class DelegatingNodeBuilder implements NodeBuilder {
    @Override
    public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
        InterruptedException {
      return getNodeBuilder().build(nodeKey, env);
    }

    @Nullable
    @Override
    public String extractTag(NodeKey nodeKey) {
      return getNodeBuilder().extractTag(nodeKey);
    }
  }

  public DelegatingNodeBuilder createDelegatingNodeBuilder() {
    return new DelegatingNodeBuilder();
  }

  /**
   * Simple node class that stores strings.
   */
  public static class StringNode implements Node {
    private final String value;

    public StringNode(String value) {
      this.value = value;
    }

    public String getValue() {
      return value;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof StringNode)) {
        return false;
      }
      return value.equals(((StringNode) o).value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }

    @Override
    public String toString() {
      return "StringNode: " + getValue();
    }
  }

  /**
   * A callback interface to provide the node computation.
   */
  public interface NodeComputer {
    /** This is called when all the declared dependencies exist. It may request new dependencies. */
    Node compute(Map<NodeKey, Node> deps, NodeBuilder.Environment env);
  }

  public static final NodeComputer COPY = new NodeComputer() {
    @Override
    public Node compute(Map<NodeKey, Node> deps, NodeBuilder.Environment env) {
      return Iterables.getOnlyElement(deps.values());
    }
  };

  public static final NodeComputer CONCATENATE = new NodeComputer() {
    @Override
    public Node compute(Map<NodeKey, Node> deps, NodeBuilder.Environment env) {
      StringBuilder result = new StringBuilder();
      for (Node node : deps.values()) {
        result.append(((StringNode) node).value);
      }
      return new StringNode(result.toString());
    }
  };
}
