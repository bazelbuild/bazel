// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.bzlmod.modcommand;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.cmdline.Label;

/** The toolchains or execution platforms that a module registers. */
public record ModuleRegistrations(ModuleKey module, ImmutableList<Registration> registrations) {

  /**
   * A target pattern passed to {@code register_toolchains} or {@code register_execution_platforms}.
   */
  public sealed interface Registration {
    /** A pattern and the targets it expands to. */
    record Expanded(String pattern, ImmutableList<RegisteredTarget> targets)
        implements Registration {}

    /** A pattern that couldn't be expanded. */
    record Failed(String pattern, String error) implements Registration {}
  }

  /**
   * A target that a registered pattern expands to. {@code mod show_execution_platforms} only
   * produces {@link Platform}s, {@code mod show_toolchains} produces all other kinds.
   */
  public sealed interface RegisteredTarget {
    /** An execution platform. */
    record Platform(Label label) implements RegisteredTarget {}

    /** A toolchain, its type and the module that owns the type. */
    record Toolchain(Label label, Label type, ModuleKey typeOwner) implements RegisteredTarget {}

    /** A target whose toolchain type couldn't be determined. */
    record Unresolved(Label label, String error) implements RegisteredTarget {}
  }

  /** Returns the number of registered patterns and targets that couldn't be analyzed. */
  public int errorCount() {
    int count = 0;
    for (var registration : registrations) {
      switch (registration) {
        case Registration.Failed _ -> count++;
        case Registration.Expanded(_, ImmutableList<RegisteredTarget> targets) -> {
          for (var target : targets) {
            if (target instanceof RegisteredTarget.Unresolved) {
              count++;
            }
          }
        }
      }
    }
    return count;
  }
}
