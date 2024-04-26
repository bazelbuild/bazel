// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands;

import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.buildtool.PathPrettyPrinter;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Encapsulates information for launching the command specified by a run invocation.
 *
 * <p>Notably, this class handles per-platform command-line formatting (windows vs unix).
 */
class RunCommandLine {
  private final ImmutableList<String> argsWithResidue;
  private final ImmutableList<String> prettyPrintArgsWithResidue;
  private final ImmutableList<String> argsWithoutResidue;
  @Nullable private final String runUnderPrefix;
  @Nullable private final String prettyRunUnderPrefix;

  private final ImmutableSortedMap<String, String> runEnvironment;
  private final Path workingDir;

  private final boolean isTestTarget;

  private RunCommandLine(
      ImmutableList<String> argsWithResidue,
      ImmutableList<String> prettyPrintArgsWithResidue,
      ImmutableList<String> argsWithoutResidue,
      @Nullable String runUnderPrefix,
      @Nullable String prettyRunUnderPrefix,
      ImmutableSortedMap<String, String> runEnvironment,
      Path workingDir,
      boolean isTestTarget) {
    this.argsWithResidue = argsWithResidue;
    this.prettyPrintArgsWithResidue = prettyPrintArgsWithResidue;
    this.argsWithoutResidue = argsWithoutResidue;
    this.runUnderPrefix = runUnderPrefix;
    this.prettyRunUnderPrefix = prettyRunUnderPrefix;
    this.runEnvironment = runEnvironment;
    this.workingDir = workingDir;
    this.isTestTarget = isTestTarget;
  }

  Path getWorkingDir() {
    return workingDir;
  }

  ImmutableSortedMap<String, String> getEnvironment() {
    return runEnvironment;
  }

  boolean isTestTarget() {
    return isTestTarget;
  }

  /**
   * Returns a console-friendly (including relative paths) representation of the command line which
   * would be returned by {@link #getArgs}.
   */
  String getPrettyArgs() {
    StringBuilder result = new StringBuilder();
    if (prettyRunUnderPrefix != null) {
      result.append(prettyRunUnderPrefix).append(" ");
    }
    for (int i = 0; i < prettyPrintArgsWithResidue.size(); i++) {
      if (i > 0) {
        result.append(" ");
      }
      result.append(ShellEscaper.escapeString(prettyPrintArgsWithResidue.get(i)));
    }
    return result.toString();
  }

  boolean requiresShExecutable() {
    return OS.getCurrent() != OS.WINDOWS || runUnderPrefix != null;
  }

  /** Returns the command arguments including residue. */
  ImmutableList<String> getArgs(String shExecutable) {
    return formatter().formatArgv(shExecutable, runUnderPrefix, argsWithResidue);
  }

  /**
   * Returns the command arguments without residue (extra arguments from the run invocation's
   * command line). This is intended to be used in places where we don't want to include the residue
   * in case it contains sensitive information.
   */
  ImmutableList<String> getArgsWithoutResidue(@Nullable String shExecutable) {
    return formatter().formatArgv(shExecutable, runUnderPrefix, argsWithoutResidue);
  }

  /**
   * Returns the script form of the command, to be used as the contents of output file in
   * --script_path mode.
   */
  String getScriptForm(String shExecutable, ImmutableSortedSet<String> environmentVarsToUnset) {
    return formatter()
        .getScriptForm(
            shExecutable,
            workingDir.getPathString(),
            environmentVarsToUnset,
            runEnvironment,
            runUnderPrefix,
            argsWithResidue);
  }

  private static Formatter formatter() {
    return OS.getCurrent() == OS.WINDOWS ? new WindowsFormatter() : new LinuxFormatter();
  }

  private interface Formatter {
    ImmutableList<String> formatArgv(
        @Nullable String shExecutable, @Nullable String runUnderPrefix, ImmutableList<String> args);

    String getScriptForm(
        String shExecutable,
        String workingDir,
        ImmutableSortedSet<String> environmentVarsToUnset,
        ImmutableSortedMap<String, String> environment,
        @Nullable String runUnderPrefix,
        ImmutableList<String> args);
  }

  @VisibleForTesting
  static class LinuxFormatter implements Formatter {
    @Override
    public ImmutableList<String> formatArgv(
        @Nullable String shExecutable,
        @Nullable String runUnderPrefix,
        ImmutableList<String> args) {
      Preconditions.checkArgument(shExecutable != null, "shExecutable must be non-null");
      StringBuilder command = new StringBuilder();
      if (runUnderPrefix != null) {
        command.append(runUnderPrefix).append(" ");
      }
      for (int i = 0; i < args.size(); i++) {
        if (i > 0) {
          command.append(" ");
        }
        command.append(ShellEscaper.escapeString(args.get(i)));
      }
      return ImmutableList.of(shExecutable, "-c", command.toString());
    }

    @Override
    public String getScriptForm(
        String shExecutable,
        String workingDir,
        ImmutableSortedSet<String> environmentVarsToUnset,
        ImmutableSortedMap<String, String> environment,
        @Nullable String runUnderPrefix,
        ImmutableList<String> args) {
      String unsetEnv =
          environmentVarsToUnset.stream().map(v -> "-u " + v).collect(joining(" \\\n    "));
      String setEnv =
          environment.entrySet().stream()
              .map(
                  kv ->
                      ShellEscaper.escapeString(kv.getKey())
                          + "="
                          + ShellEscaper.escapeString(kv.getValue()))
              .collect(joining(" \\\n    "));
      String commandLine = getCommandLine(shExecutable, runUnderPrefix, args);

      StringBuilder result = new StringBuilder();
      result.append("#!").append(shExecutable).append("\n");
      result.append("cd ").append(ShellEscaper.escapeString(workingDir)).append(" && \\\n");
      result.append("  exec env \\\n");
      result.append("    ").append(unsetEnv).append(" \\\n");
      result.append("    ").append(setEnv).append(" \\\n");
      result.append("  ").append(commandLine).append(" \"$@\"");

      return result.toString();
    }

    private static String getCommandLine(
        String shExecutable, @Nullable String runUnderPrefix, ImmutableList<String> args) {
      StringBuilder command = new StringBuilder();
      if (runUnderPrefix != null) {
        command.append(runUnderPrefix).append(" ");
      }
      for (int i = 0; i < args.size(); i++) {
        if (i > 0) {
          command.append(" ");
        }
        command.append(ShellEscaper.escapeString(args.get(i)));
      }

      if (runUnderPrefix == null) {
        return command.toString();
      } else {
        return shExecutable + " -c " + ShellEscaper.escapeString(command.toString());
      }
    }
  }

  @VisibleForTesting
  static class WindowsFormatter implements Formatter {
    @Override
    public ImmutableList<String> formatArgv(
        @Nullable String shExecutable,
        @Nullable String runUnderPrefix,
        ImmutableList<String> args) {
      if (runUnderPrefix != null) {
        Preconditions.checkArgument(
            shExecutable != null, "shExecutable must be non-null when --run_under is used");
        StringBuilder command = new StringBuilder();
        command.append(runUnderPrefix).append(" ");
        for (int i = 0; i < args.size(); i++) {
          if (i > 0) {
            command.append(" ");
          }
          command.append(ShellEscaper.escapeString(args.get(i)));
        }
        return ImmutableList.of(
            shExecutable, "-c", ShellUtils.windowsEscapeArg(command.toString()));
      }

      ImmutableList.Builder<String> result = ImmutableList.builder();
      for (int i = 0; i < args.size(); i++) {
        if (i == 0) {
          // All but the first element in `cmdLine` have to be escaped. The first element is the
          // binary, which must not be escaped.
          result.add(args.get(i));
        } else {
          result.add(ShellUtils.windowsEscapeArg(args.get(i)));
        }
      }
      return result.build();
    }

    @Override
    public String getScriptForm(
        String shExecutable,
        String workingDir,
        ImmutableSortedSet<String> environmentVarsToUnset,
        ImmutableSortedMap<String, String> environment,
        @Nullable String runUnderPrefix,
        ImmutableList<String> args) {

      String unsetEnv =
          environmentVarsToUnset.stream().map(v -> "SET " + v + "=").collect(joining("\n  "));
      String setEnv =
          environment.entrySet().stream()
              .map(kv -> "SET " + kv.getKey() + "=" + kv.getValue())
              .collect(joining("\n  "));
      String commandLine = getCommandLine(shExecutable, runUnderPrefix, args);

      StringBuilder result = new StringBuilder();
      result.append("@echo off\n");
      result.append("cd /d ").append(workingDir).append("\n");
      result.append("  ").append(unsetEnv).append("\n");
      result.append("  ").append(setEnv).append("\n");
      result.append("  ").append(commandLine).append(" %*");
      return result.toString();
    }

    private static String getCommandLine(
        String shExecutable, @Nullable String runUnderPrefix, ImmutableList<String> args) {
      StringBuilder command = new StringBuilder();
      if (runUnderPrefix != null) {
        command.append(runUnderPrefix).append(" ");
      }
      for (int i = 0; i < args.size(); i++) {
        if (i == 0) {
          command.append(args.get(i).replace('/', '\\'));
        } else {
          command.append(" ").append(ShellUtils.windowsEscapeArg(args.get(i)));
        }
      }
      if (runUnderPrefix == null) {
        return command.toString();
      } else {
        return shExecutable + " -c " + ShellEscaper.escapeString(command.toString());
      }
    }
  }

  static class Builder {
    private final ImmutableSortedMap<String, String> runEnvironment;
    private final Path workingDir;
    private final boolean isTestTarget;

    @Nullable private String runUnderPrefix;
    @Nullable private String prettyRunUnderPrefix;

    private final ImmutableList.Builder<String> args = ImmutableList.builder();
    private final ImmutableList.Builder<String> prettyPrintArgs = ImmutableList.builder();
    private final ImmutableList.Builder<String> residueArgs = ImmutableList.builder();

    Builder(
        ImmutableSortedMap<String, String> runEnvironment, Path workingDir, boolean isTestTarget) {
      this.runEnvironment = runEnvironment;
      this.workingDir = workingDir;
      this.isTestTarget = isTestTarget;
    }

    @CanIgnoreReturnValue
    Builder setRunUnderPrefix(String runUnderPrefix) {
      this.runUnderPrefix = runUnderPrefix;
      this.prettyRunUnderPrefix = runUnderPrefix;
      return this;
    }

    @CanIgnoreReturnValue
    Builder setRunUnderTarget(
        Path runUnderBinary, List<String> args, PathPrettyPrinter pathPrettyPrinter) {
      StringBuilder runUnder = new StringBuilder();
      StringBuilder prettyRunUnder = new StringBuilder();
      runUnder.append(ShellEscaper.escapeString(runUnderBinary.getPathString()));
      prettyRunUnder.append(
          ShellEscaper.escapeString(
              pathPrettyPrinter.getPrettyPath(runUnderBinary.asFragment()).getPathString()));
      for (String arg : args) {
        String escapedArg = ShellEscaper.escapeString(arg);
        runUnder.append(" ").append(escapedArg);
        prettyRunUnder.append(" ").append(escapedArg);
      }
      this.runUnderPrefix = runUnder.toString();
      this.prettyRunUnderPrefix = prettyRunUnder.toString();
      return this;
    }

    @CanIgnoreReturnValue
    Builder addArg(String arg) {
      return addArgInternal(arg, arg);
    }

    @CanIgnoreReturnValue
    Builder addArg(Path path, PathPrettyPrinter pathPrettyPrinter) {
      return addArgInternal(
          path.getPathString(), pathPrettyPrinter.getPrettyPath(path.asFragment()).getPathString());
    }

    @CanIgnoreReturnValue
    Builder addArgs(Iterable<String> args) {
      for (String arg : args) {
        addArg(arg);
      }
      return this;
    }

    @CanIgnoreReturnValue
    Builder addArgsFromResidue(ImmutableList<String> args) {
      residueArgs.addAll(args);
      return this;
    }

    @CanIgnoreReturnValue
    private Builder addArgInternal(String arg, String prettyPrintArg) {
      args.add(arg);
      prettyPrintArgs.add(prettyPrintArg);
      return this;
    }

    RunCommandLine build() {
      ImmutableList<String> argsWithoutResidue = args.build();
      ImmutableList<String> argsWithResidue =
          ImmutableList.<String>builder()
              .addAll(argsWithoutResidue)
              .addAll(residueArgs.build())
              .build();
      ImmutableList<String> prettyPrintArgsWithResidue =
          ImmutableList.<String>builder()
              .addAll(prettyPrintArgs.build())
              .addAll(residueArgs.build())
              .build();
      return new RunCommandLine(
          argsWithResidue,
          prettyPrintArgsWithResidue,
          argsWithoutResidue,
          runUnderPrefix,
          prettyRunUnderPrefix,
          runEnvironment,
          workingDir,
          isTestTarget);
    }
  }
}
