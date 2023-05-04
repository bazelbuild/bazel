// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Memory-efficient {@link CommandLine} for a shell command.
 *
 * <p>Equivalent to invoking {@link #shExecutable} (e.g. {@code /bin/bash}) followed by {@code -c}
 * and {@link #command}. Supports optionally padding the command line with an empty argument, which
 * can be useful to ensure that any subsequent arguments get assigned to {@code $1} etc.
 */
final class ShellCommand extends CommandLine {

  private final PathFragment shExecutable;
  private final String command;
  private final boolean pad;

  ShellCommand(PathFragment shExecutable, String command, boolean pad) {
    this.shExecutable = checkNotNull(shExecutable);
    this.command = checkNotNull(command);
    this.pad = pad;
  }

  @Override
  public ImmutableList<String> arguments() {
    return pad
        ? ImmutableList.of(shExecutable.expandToCommandLine(), "-c", command, "")
        : ImmutableList.of(shExecutable.expandToCommandLine(), "-c", command);
  }
}
