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
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsCycle;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsExpanded;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.IsIndirect;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor.ResultNode.NodeMetadata;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryOptions.Charset;
import com.google.devtools.build.lib.bazel.bzlmod.modquery.OutputFormatters.OutputFormatter;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Objects;

/**
 * Outputs graph-based results of {@link
 * com.google.devtools.build.lib.bazel.bzlmod.modquery.ModqueryExecutor} in a human-readable text
 * format.
 */
public class TextOutputFormatter extends OutputFormatter {

  private Deque<Boolean> isLastChildStack;
  private DrawCharset drawCharset;

  @Override
  public void output() {
    if (options.charset == Charset.ASCII) {
      drawCharset = DrawCharset.ASCII;
    } else {
      drawCharset = DrawCharset.UTF8;
    }
    isLastChildStack = new ArrayDeque<>();
    printTree(ModuleKey.ROOT, null, IsExpanded.TRUE, IsIndirect.FALSE, IsCycle.FALSE, 0);
  }

  // Depth-first traversal to print the actual output
  void printTree(
      ModuleKey key,
      ModuleKey parent,
      IsExpanded expanded,
      IsIndirect indirect,
      IsCycle cycle,
      int depth) {
    ResultNode node = Objects.requireNonNull(result.get(key));
    StringBuilder str = new StringBuilder();

    if (depth > 0) {
      int indents = isLastChildStack.size() - 1;
      Iterator<Boolean> value = isLastChildStack.descendingIterator();
      for (int i = 0; i < indents; i++) {
        boolean isLastChild = value.next();
        if (isLastChild) {
          str.append(drawCharset.emptyIndent);
        } else {
          str.append(drawCharset.prevChildIndent);
        }
      }
      if (indirect == IsIndirect.TRUE) {
        if (isLastChildStack.getFirst()) {
          str.append(drawCharset.lastIndirectChildIndent);
        } else {
          str.append(drawCharset.indirectChildIndent);
        }
      } else {
        if (isLastChildStack.getFirst()) {
          str.append(drawCharset.lastChildIndent);

        } else {
          str.append(drawCharset.childIndent);
        }
      }
    }

    int totalChildrenNum = node.getChildren().size();

    if (key.equals(ModuleKey.ROOT)) {
      AugmentedModule rootModule = depGraph.get(ModuleKey.ROOT);
      Preconditions.checkNotNull(rootModule);
      str.append(
          String.format(
              "root (%s@%s)",
              rootModule.getName(),
              rootModule.getVersion().equals(Version.EMPTY) ? "_" : rootModule.getVersion()));
    } else {
      str.append(key).append(" ");
    }

    if (cycle == IsCycle.TRUE) {
      str.append("(cycle) ");
    } else if (expanded == IsExpanded.FALSE) {
      str.append("(*) ");
    } else {
      if (totalChildrenNum != 0 && node.isTarget()) {
        str.append("# ");
      }
    }
    AugmentedModule module = Objects.requireNonNull(depGraph.get(key));

    if (!options.extra && !module.isUsed()) {
      str.append("(unused) ");
    }
    // If the edge is indirect, the parent is not only unknown, but the node could have come
    // from
    // multiple paths merged in the process, so we skip the resolution explanation.
    if (indirect == IsIndirect.FALSE && options.extra && parent != null) {
      Explanation explanation = getExtraResolutionExplanation(key, parent);
      if (explanation != null) {
        str.append(explanation.toExplanationString(!module.isUsed()));
      }
    }

    this.printer.println(str);

    if (expanded == IsExpanded.FALSE) {
      return;
    }

    int currChild = 1;
    for (Entry<ModuleKey, NodeMetadata> e : node.getChildrenSortedByEdgeType()) {
      ModuleKey childKey = e.getKey();
      IsExpanded childExpanded = e.getValue().isExpanded();
      IsIndirect childIndirect = e.getValue().isIndirect();
      IsCycle childCycles = e.getValue().isCycle();
      isLastChildStack.push(currChild++ == totalChildrenNum);
      printTree(childKey, key, childExpanded, childIndirect, childCycles, depth + 1);
      isLastChildStack.pop();
    }
  }

  enum DrawCharset {
    ASCII("    ", "|   ", "|___", "|...", "|___", "|..."),
    UTF8("    ", "│   ", "├───", "├╌╌╌", "└───", "└╌╌╌");
    final String emptyIndent;
    final String prevChildIndent;
    final String childIndent;
    final String indirectChildIndent;
    final String lastChildIndent;
    final String lastIndirectChildIndent;

    DrawCharset(
        String emptyIndent,
        String prevChildIndent,
        String childIndent,
        String indirectChildIndent,
        String lastChildIndent,
        String lastIndirectChildIndent) {
      this.emptyIndent = emptyIndent;
      this.prevChildIndent = prevChildIndent;
      this.childIndent = childIndent;
      this.indirectChildIndent = indirectChildIndent;
      this.lastChildIndent = lastChildIndent;
      this.lastIndirectChildIndent = lastIndirectChildIndent;
    }
  }
}
