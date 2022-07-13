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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleInspectorValue.AugmentedModule.ResolutionReason;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter;
import com.google.devtools.build.lib.util.io.AnsiTerminalPrinter.Mode;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

/**
 * Executes inspection queries for {@link
 * com.google.devtools.build.lib.bazel.commands.ModqueryCommand} and prints the resulted output.
 */
public class ModqueryExecutor {
  private final ImmutableMap<ModuleKey, Module> resolvedDepGraph;
  private final ImmutableMap<ModuleKey, AugmentedModule> depGraph;
  private final ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex;
  private final AnsiTerminalPrinter printer;

  public ModqueryExecutor(
      ImmutableMap<ModuleKey, Module> resolvedDepGraph,
      ImmutableMap<ModuleKey, AugmentedModule> depGraph,
      ImmutableMap<String, ImmutableSet<ModuleKey>> modulesIndex,
      AnsiTerminalPrinter printer) {
    this.resolvedDepGraph = resolvedDepGraph;
    this.depGraph = depGraph;
    this.modulesIndex = modulesIndex;
    this.printer = printer;
  }

  public void tree(ImmutableSet<ModuleKey> from) {
    printer.printLn(Mode.WARNING + "DUMMY IMPLEMENTATION" + Mode.DEFAULT);
    printer.printLn("");
    printer.printLn("All modules index:");
    printer.printLn(modulesIndex.toString());
    printer.printLn("");
    for (ModuleKey target : from) {
      printer.printLn(Mode.INFO.toString() + target + Mode.DEFAULT);
      Set<ModuleKey> seen = new HashSet<>();
      Deque<ModuleKey> toVisit = new ArrayDeque<>();
      toVisit.add(target);
      seen.add(target);
      while (!toVisit.isEmpty()) {
        ModuleKey curr = toVisit.remove();
        AugmentedModule module = depGraph.get(curr);
        for (Entry<ModuleKey, ResolutionReason> e : module.getDeps().entrySet()) {
          ModuleKey child = e.getKey();
          if (!resolvedDepGraph.containsKey(child) || seen.contains(child)) {
            continue;
          }
          seen.add(child);
          toVisit.add(child);
          printer.printLn(child + " " + e.getValue());
        }
      }
      printer.printLn("");
    }
  }

  public void deps(ImmutableSet<ModuleKey> targets) {
    printer.printLn(Mode.WARNING + "DUMMY IMPLEMENTATION" + Mode.DEFAULT);
    printer.printLn("");
    for (ModuleKey target : targets) {
      printer.printLn(Mode.INFO.toString() + target + Mode.DEFAULT);
      for (Entry<ModuleKey, ResolutionReason> e : depGraph.get(target).getDeps().entrySet()) {
        printer.printLn(e.getKey() + " " + e.getValue());
      }
      printer.printLn("");
    }
  }

  public void path(ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void allPaths(ImmutableSet<ModuleKey> from, ImmutableSet<ModuleKey> to) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void explain(ImmutableSet<ModuleKey> targets) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public void show(ImmutableSet<ModuleKey> targets) {
    throw new UnsupportedOperationException("Not implemented yet");
  }
}
