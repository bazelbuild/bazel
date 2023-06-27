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

import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsCycle;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsExpanded;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsIndirect;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.NodeMetadata;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.OutputFormatters.OutputFormatter;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.Map.Entry;

/** Outputs graph-based results of {@link ModqueryExecutor} in JSON format. */
public class JsonOutputFormatter extends OutputFormatter {
  @Override
  public void output() {
    JsonObject root = printTree(ModuleKey.ROOT, null, IsExpanded.TRUE, IsIndirect.FALSE);
    root.addProperty("root", true);
    printer.println(new GsonBuilder().setPrettyPrinting().create().toJson(root));
  }

  public String printKey(ModuleKey key) {
    if (key.equals(ModuleKey.ROOT)) {
      return "root";
    }
    return key.toString();
  }

  JsonObject printTree(ModuleKey key, ModuleKey parent, IsExpanded expanded, IsIndirect indirect) {
    ResultNode node = result.get(key);
    AugmentedModule module = depGraph.get(key);
    JsonObject json = new JsonObject();
    json.addProperty("key", printKey(key));
    if (!key.getName().equals(module.getName())) {
      json.addProperty("name", module.getName());
    }
    if (!key.getVersion().equals(module.getVersion())) {
      json.addProperty("version", module.getVersion().toString());
    }

    if (indirect == IsIndirect.FALSE && options.extra && parent != null) {
      Explanation explanation = getExtraResolutionExplanation(key, parent);
      if (explanation != null) {
        if (!module.isUsed()) {
          json.addProperty("unused", true);
          json.addProperty("resolvedVersion", explanation.getChangedVersion().toString());
        } else {
          json.addProperty("originalVersion", explanation.getChangedVersion().toString());
        }
        json.addProperty("resolutionReason", explanation.getChangedVersion().toString());
        if (explanation.getRequestedByModules() != null) {
          JsonArray requestedBy = new JsonArray();
          explanation.getRequestedByModules().forEach(k -> requestedBy.add(printKey(k)));
          json.add("resolvedRequestedBy", requestedBy);
        }
      }
    }

    if (expanded == IsExpanded.FALSE) {
      json.addProperty("unexpanded", true);
      return json;
    }

    JsonArray deps = new JsonArray();
    JsonArray indirectDeps = new JsonArray();
    JsonArray cycles = new JsonArray();
    for (Entry<ModuleKey, NodeMetadata> e : node.getChildrenSortedByEdgeType()) {
      ModuleKey childKey = e.getKey();
      IsExpanded childExpanded = e.getValue().isExpanded();
      IsIndirect childIndirect = e.getValue().isIndirect();
      IsCycle childCycles = e.getValue().isCycle();
      if (childCycles == IsCycle.TRUE) {
        cycles.add(printTree(childKey, key, IsExpanded.FALSE, IsIndirect.FALSE));
      } else if (childIndirect == IsIndirect.TRUE) {
        indirectDeps.add(printTree(childKey, key, childExpanded, IsIndirect.TRUE));
      } else {
        deps.add(printTree(childKey, key, childExpanded, IsIndirect.FALSE));
      }
    }
    json.add("dependencies", deps);
    json.add("indirectDependencies", indirectDeps);
    json.add("cycles", cycles);
    return json;
  }
}
