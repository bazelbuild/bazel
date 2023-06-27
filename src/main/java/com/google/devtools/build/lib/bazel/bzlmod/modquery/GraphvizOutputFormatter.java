// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod.modquery;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsIndirect;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.NodeMetadata;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.OutputFormatters.OutputFormatter;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Outputs graph-based results of {@link ModqueryExecutor} in the Graphviz <i>dot</i> format which
 * can be further pipelined to create an image graph visualization.
 */
public class GraphvizOutputFormatter extends OutputFormatter {

  @Override
  public void output() {
    StringBuilder str = new StringBuilder();
    str.append("digraph mygraph {\n")
        .append("  ")
        .append("node [ shape=box ]\n")
        .append("  ")
        .append("edge [ fontsize=8 ]\n");
    Set<ModuleKey> seen = new HashSet<>();
    Deque<ModuleKey> toVisit = new ArrayDeque<>();
    seen.add(ModuleKey.ROOT);
    toVisit.add(ModuleKey.ROOT);

    while (!toVisit.isEmpty()) {
      ModuleKey key = toVisit.pop();
      AugmentedModule module = depGraph.get(key);
      ResultNode node = result.get(key);
      Preconditions.checkNotNull(module);
      Preconditions.checkNotNull(node);
      String sourceId = toId(key);

      if (key.equals(ModuleKey.ROOT)) {
        String rootLabel = String.format("root (%s@%s)", module.getName(), module.getVersion());
        str.append(String.format("  root [ label=\"%s\" ]\n", rootLabel));
      } else if (node.isTarget() || !module.isUsed()) {
        String shapeString = node.isTarget() ? "diamond" : "box";
        String styleString = module.isUsed() ? "solid" : "dotted";
        str.append(
            String.format("  %s [ shape=%s style=%s ]\n", toId(key), shapeString, styleString));
      }

      for (Entry<ModuleKey, NodeMetadata> e : node.getChildrenSortedByKey()) {
        ModuleKey childKey = e.getKey();
        IsIndirect childIndirect = e.getValue().isIndirect();
        String childId = toId(childKey);
        if (childIndirect == IsIndirect.FALSE) {
          String reasonLabel = getReasonLabel(childKey, key);
          str.append(String.format("  %s -> %s [ %s ]\n", sourceId, childId, reasonLabel));
        } else {
          str.append(String.format("  %s -> %s [ style=dashed ]\n", sourceId, childId));
        }
        if (seen.add(childKey)) {
          toVisit.add(childKey);
        }
      }
    }
    str.append("}");
    printer.println(str);
    printer.flush();
  }

  private String toId(ModuleKey key) {
    if (key.equals(ModuleKey.ROOT)) {
      return "root";
    }
    return String.format(
        "\"%s@%s\"",
        key.getName(), key.getVersion().equals(Version.EMPTY) ? "_" : key.getVersion());
  }

  private String getReasonLabel(ModuleKey key, ModuleKey parent) {
    if (!options.extra) {
      return "";
    }
    Explanation explanation = getExtraResolutionExplanation(key, parent);
    if (explanation == null) {
      return "";
    }
    String label = explanation.getResolutionReason().getLabel();
    if (!label.isEmpty()) {
      return String.format("label=%s", label);
    }
    return "";
  }
}
