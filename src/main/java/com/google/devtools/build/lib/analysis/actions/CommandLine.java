// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.util.Preconditions;

/** A representation of a list of arguments, often a command executed by {@link SpawnAction}. */
public abstract class CommandLine {
  /**
   * Returns the command line.
   */
  public abstract Iterable<String> arguments();

  /**
   * Returns the evaluated command line with enclosed artifacts expanded by {@code artifactExpander}
   * at execution time.
   *
   * <p>By default, this method just delegates to {@link #arguments()}, without performing any
   * artifact expansion. Subclasses should override this method if they contain TreeArtifacts and
   * need to expand them for proper argument evaluation.
   */
  public Iterable<String> arguments(ArtifactExpander artifactExpander) {
    return arguments();
  }

  /** Returns a {@link CommandLine} backed by a copy of the given list of arguments. */
  public static CommandLine of(Iterable<String> arguments) {
    final Iterable<String> immutableArguments = CollectionUtils.makeImmutable(arguments);
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        return immutableArguments;
      }
    };
  }

  /**
   * Returns a {@link CommandLine} that is constructed by prepending the {@code executableArgs} to
   * {@code commandLine}.
   */
  CommandLine prepend(final ImmutableList<String> executableArgs) {
    final CommandLine self = this;
    Preconditions.checkState(!executableArgs.isEmpty());
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        return Iterables.concat(executableArgs, self.arguments());
      }

      @Override
      public Iterable<String> arguments(ArtifactExpander artifactExpander) {
        return Iterables.concat(executableArgs, self.arguments(artifactExpander));
      }
    };
  }

  /**
   * This helps when debugging Blaze code that uses {@link CommandLine}s, as you can see their
   * content directly in the variable inspector.
   */
  @Override
  public String toString() {
    return Joiner.on(' ').join(arguments());
  }
}
