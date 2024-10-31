// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.output;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Utility class to hold all conditional edges in a graph. Allows easy look-up of all conditions
 * between two nodes.
 */
public class ConditionalEdges {
  // Map containing all the conditions for all the conditional edges in a graph.
  private HashMap<
          Label /* src node */, SetMultimap<Label /* dest node */, Label /* condition labels */>>
      map;

  public ConditionalEdges() {}

  /** Builds ConditionalEdges from given graph. */
  public ConditionalEdges(Digraph<Target> graph) {
    this.map = new HashMap<>();

    for (Node<Target> node : graph.getNodes()) {
      Rule rule = node.getLabel().getAssociatedRule();
      if (rule == null) {
        // rule is null for source files and package groups. Skip them.
        continue;
      }

      SetMultimap<Label, Label> conditions = getAllConditions(rule, RawAttributeMapper.of(rule));
      if (conditions.isEmpty()) {
        // bail early for most common case of no conditions in the rule.
        continue;
      }

      Label nodeLabel = node.getLabel().getLabel();
      for (Node<Target> succ : node.getSuccessors()) {
        Label successorLabel = succ.getLabel().getLabel();
        if (conditions.containsKey(successorLabel)) {
          insert(nodeLabel, successorLabel, conditions.get(successorLabel));
        }
      }
    }
  }

  /** Inserts `conditions` for edge src --> dest. */
  public void insert(Label src, Label dest, Set<Label> conditions) {
    map.computeIfAbsent(src, (k) -> HashMultimap.create());
    map.get(src).putAll(dest, conditions);
  }

  /**
   * Returns all conditions for edge src --> dest, if they exist. Does not return default
   * conditions.
   */
  public Optional<Set<Label>> get(Label src, Label dest) {
    if (!map.containsKey(src) || !map.get(src).containsKey(dest)) {
      return Optional.empty();
    }

    return Optional.of(map.get(src).get(dest));
  }

  /**
   * Returns map of dependency to list of condition-labels.
   *
   * <p>Example: For a rule like below,
   *
   * <pre>
   *  some_rule(
   *    ...
   *    deps = [
   *      ... default dependencies ...
   *    ] + select ({
   *      "//some:config1": [ "//some:a", "//some:common" ],
   *      "//some:config2": [ "//other:a", "//some:common" ],
   *      "//conditions:default": [ "//some:default" ],
   *    })
   *  )
   * </pre>
   *
   * it returns following map:
   *
   * <pre>
   *  {
   *    "//some:a": ["//some:config1" ]
   *    "//other:a": ["//some:config2" ]
   *    "//some:common": ["//some:config1", "//some:config2" ]
   *    "//some:default": [ "//conditions:default" ]
   *  }
   * </pre>
   */
  private SetMultimap<Label, Label> getAllConditions(Rule rule, RawAttributeMapper attributeMap) {
    SetMultimap<Label, Label> conditions = HashMultimap.create();
    for (Attribute attr : rule.getAttributes()) {
      // TODO(bazel-team): Handle the case where dependency exists through both configurable as well
      // as non-configurable attributes. Currently this prints such an edge as a conditional one.
      if (!attributeMap.isConfigurable(attr.getName())) {
        // skip non configurable attributes
        continue;
      }
      if (rule.getAttr(attr.getName()) instanceof Attribute.ComputedDefault) {
        // isConfigurable above checks that the attribute is either a `select()` or a computed
        // default. We don't currently handle the latter so skip it.
        // TODO: b/375344172 - (bazel-team) Decide how to resolve computed defaults.
        // TODO: b/375344172 - (bazel-team) Add a regression test for this case.
        continue;
      }

      for (BuildType.Selector<?> selector :
          ((BuildType.SelectorList<?>) attributeMap.getRawAttributeValue(rule, attr))
              .getSelectors()) {
        if (selector.isUnconditional()) {
          // skip unconditional selectors
          continue;
        }
        selector.forEach(
            (key, value) -> {
              if (value instanceof List<?> deps) {
                for (Object dep : deps) {
                  if (dep instanceof Label label) {
                    conditions.put(label, key);
                  }
                }
              } else if (value instanceof Label label) {
                conditions.put(label, key);
              }
            });
      }
    }
    return conditions;
  }
}
