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
import java.util.Map.Entry;

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

  public void deps(ModuleKey target) {
    for (Entry<ModuleKey, ResolutionReason> e : depGraph.get(target).getDeps().entrySet()) {
      printer.printLn(e.getKey() + " " + e.getValue().toString());
    }
  }

  public void transitiveDeps(ModuleKey target) {}

  public void path(ModuleKey from, ModuleKey to) {}

  public void allPaths(ModuleKey from, ModuleKey to) {}

  public void explain(ImmutableSet<ModuleKey> targets) {}
}
