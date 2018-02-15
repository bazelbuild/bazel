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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.Strategy;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.util.Fingerprint;

/** A representation of a list of arguments. */
@AutoCodec(strategy = Strategy.POLYMORPHIC)
public abstract class CommandLine {
  public static final ObjectCodec<CommandLine> CODEC = new CommandLine_AutoCodec();

  @AutoCodec
  @VisibleForSerialization
  static class EmptyCommandLine extends CommandLine {
    public static final ObjectCodec<EmptyCommandLine> CODEC =
        new CommandLine_EmptyCommandLine_AutoCodec();

    @Override
    public Iterable<String> arguments() throws CommandLineExpansionException {
      return ImmutableList.of();
    }
  }

  public static final CommandLine EMPTY = new EmptyCommandLine();

  /** Returns the command line. */
  public abstract Iterable<String> arguments() throws CommandLineExpansionException;

  /**
   * Returns the evaluated command line with enclosed artifacts expanded by {@code artifactExpander}
   * at execution time.
   *
   * <p>By default, this method just delegates to {@link #arguments()}, without performing any
   * artifact expansion. Subclasses should override this method if they contain TreeArtifacts and
   * need to expand them for proper argument evaluation.
   */
  public Iterable<String> arguments(ArtifactExpander artifactExpander)
      throws CommandLineExpansionException {
    return arguments();
  }

  public void addToFingerprint(ActionKeyContext actionKeyContext, Fingerprint fingerprint)
      throws CommandLineExpansionException {
    for (String s : arguments()) {
      fingerprint.addString(s);
    }
  }

  @AutoCodec
  @VisibleForSerialization
  static class ArgumentCommandLine extends CommandLine {
    public static final ObjectCodec<ArgumentCommandLine> CODEC =
        new CommandLine_ArgumentCommandLine_AutoCodec();

    private Iterable<String> args;

    ArgumentCommandLine(Iterable<String> args) {
      this.args = args;
    }

    @Override
    public Iterable<String> arguments() throws CommandLineExpansionException {
      return args;
    }
  }

  /** Returns a {@link CommandLine} backed by a copy of the given list of arguments. */
  public static CommandLine of(Iterable<String> arguments) {
    final Iterable<String> immutableArguments = CollectionUtils.makeImmutable(arguments);
    return new ArgumentCommandLine(immutableArguments);
  }

  @AutoCodec
  @VisibleForSerialization
  static class ConcatenatedCommandLine extends CommandLine {
    public static final ObjectCodec<ConcatenatedCommandLine> CODEC =
        new CommandLine_ConcatenatedCommandLine_AutoCodec();

    private ImmutableList<String> executableArgs;
    private CommandLine commandLine;

    @VisibleForSerialization
    ConcatenatedCommandLine(ImmutableList<String> executableArgs, CommandLine commandLine) {
      this.executableArgs = executableArgs;
      this.commandLine = commandLine;
    }

    @Override
    public Iterable<String> arguments() throws CommandLineExpansionException {
      return Iterables.concat(executableArgs, commandLine.arguments());
    }

    @Override
    public Iterable<String> arguments(ArtifactExpander artifactExpander)
        throws CommandLineExpansionException {
      return Iterables.concat(executableArgs, commandLine.arguments(artifactExpander));
    }
  }

  /**
   * Returns a {@link CommandLine} that is constructed by prepending the {@code executableArgs} to
   * {@code commandLine}.
   */
  public static CommandLine concat(
      final ImmutableList<String> executableArgs, final CommandLine commandLine) {
    if (executableArgs.isEmpty()) {
      return commandLine;
    }
    return new ConcatenatedCommandLine(executableArgs, commandLine);
  }

  @AutoCodec
  @VisibleForSerialization
  static class ReverseConcatenatedCommandLine extends CommandLine {
    public static final ObjectCodec<ReverseConcatenatedCommandLine> CODEC =
        new CommandLine_ReverseConcatenatedCommandLine_AutoCodec();

    private ImmutableList<String> executableArgs;
    private CommandLine commandLine;

    @VisibleForSerialization
    ReverseConcatenatedCommandLine(ImmutableList<String> executableArgs, CommandLine commandLine) {
      this.executableArgs = executableArgs;
      this.commandLine = commandLine;
    }

    @Override
    public Iterable<String> arguments() throws CommandLineExpansionException {
      return Iterables.concat(commandLine.arguments(), executableArgs);
    }

    @Override
    public Iterable<String> arguments(ArtifactExpander artifactExpander)
        throws CommandLineExpansionException {
      return Iterables.concat(commandLine.arguments(artifactExpander), executableArgs);
    }
  }

  /**
   * Returns a {@link CommandLine} that is constructed by appending the {@code args} to {@code
   * commandLine}.
   */
  public static CommandLine concat(
      final CommandLine commandLine, final ImmutableList<String> args) {
    if (args.isEmpty()) {
      return commandLine;
    }
    return new ReverseConcatenatedCommandLine(args, commandLine);
  }

  /**
   * This helps when debugging Blaze code that uses {@link CommandLine}s, as you can see their
   * content directly in the variable inspector.
   */
  @Override
  public String toString() {
    try {
      return Joiner.on(' ').join(arguments());
    } catch (CommandLineExpansionException e) {
      return "Error in expanding command line";
    }
  }
}
