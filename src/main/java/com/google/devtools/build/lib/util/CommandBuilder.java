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

package com.google.devtools.build.lib.util;

import static com.google.common.base.StandardSystemProperty.JAVA_IO_TMPDIR;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implements OS aware {@link Command} builder. At this point only Linux, Mac and Windows XP are
 * supported.
 *
 * <p>Builder will also apply heuristic to identify trivial cases where unix-like command lines
 * could be automatically converted into the Windows-compatible form.
 *
 * <p>TODO(bazel-team): (2010) Some of the code here is very similar to the {@link
 * com.google.devtools.build.lib.shell.Shell} class. This should be looked at.
 */
public final class CommandBuilder {

  private static final ImmutableList<String> SHELLS = ImmutableList.of("/bin/sh", "/bin/bash");

  private static final Splitter ARGV_SPLITTER = Splitter.on(CharMatcher.anyOf(" \t"));

  private final OS system;
  private final ImmutableMap<String, String> clientEnv;
  private final List<String> argv = new ArrayList<>();
  private final Map<String, String> env = new HashMap<>();
  private File workingDir = null;
  private boolean useShell = false;

  public CommandBuilder(Map<String, String> clientEnv) {
    this(OS.getCurrent(), clientEnv);
  }

  @VisibleForTesting
  CommandBuilder(OS system, Map<String, String> clientEnv) {
    this.system = system;
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
  }

  @CanIgnoreReturnValue
  public CommandBuilder addArg(String arg) {
    Preconditions.checkNotNull(arg, "Argument must not be null");
    argv.add(arg);
    return this;
  }

  @CanIgnoreReturnValue
  public CommandBuilder addArgs(Iterable<String> args) {
    Preconditions.checkArgument(!Iterables.contains(args, null), "Arguments must not be null");
    Iterables.addAll(argv, args);
    return this;
  }

  public CommandBuilder addArgs(String... args) {
    return addArgs(Arrays.asList(args));
  }

  @CanIgnoreReturnValue
  public CommandBuilder addEnv(Map<String, String> env) {
    Preconditions.checkNotNull(env);
    this.env.putAll(env);
    return this;
  }

  @CanIgnoreReturnValue
  public CommandBuilder emptyEnv() {
    env.clear();
    return this;
  }

  @CanIgnoreReturnValue
  public CommandBuilder setEnv(Map<String, String> env) {
    emptyEnv();
    addEnv(env);
    return this;
  }

  @CanIgnoreReturnValue
  public CommandBuilder setWorkingDir(Path path) {
    Preconditions.checkNotNull(path);
    workingDir = path.getPathFile();
    return this;
  }

  @CanIgnoreReturnValue
  public CommandBuilder useTempDir() {
    workingDir = new File(JAVA_IO_TMPDIR.value());
    return this;
  }

  @CanIgnoreReturnValue
  public CommandBuilder useShell(boolean useShell) {
    this.useShell = useShell;
    return this;
  }

  private boolean argvStartsWithSh() {
    return argv.size() >= 2 && SHELLS.contains(argv.get(0)) && "-c".equals(argv.get(1));
  }

  private String[] transformArgvForLinux() {
    // If command line already starts with "/bin/sh -c", ignore useShell attribute.
    if (useShell && !argvStartsWithSh()) {
      // c.g.io.base.shell.Shell.shellify() actually concatenates argv into the space-separated
      // string here. Not sure why, but we will do the same.
      return new String[] {"/bin/sh", "-c", Joiner.on(' ').join(argv)};
    }
    return argv.toArray(new String[argv.size()]);
  }

  private String[] transformArgvForWindows() {
    List<String> modifiedArgv;
    // Heuristic: replace "/bin/sh -c" with something more appropriate for Windows.
    if (argvStartsWithSh()) {
      useShell = true;
      modifiedArgv = Lists.newArrayList(argv.subList(2, argv.size()));
    } else {
      modifiedArgv = Lists.newArrayList(argv);
    }

    if (!modifiedArgv.isEmpty()) {
      // args can contain whitespace, so figure out the first word
      String argv0 = modifiedArgv.get(0);
      String command = ARGV_SPLITTER.split(argv0).iterator().next();

      // Automatically enable CMD.EXE use if we are executing something else besides "*.exe" file.
      // When use CMD.EXE to invoke a bat/cmd file, the file path must have '\' instead of '/'
      if (!command.toLowerCase().endsWith(".exe")) {
        useShell = true;
        modifiedArgv.set(0, argv0.replace('/', '\\'));
      }
    } else {
      // This is degenerate "/bin/sh -c" case. We ensure that Windows behavior is identical
      // to the Linux - call shell that will do nothing.
      useShell = true;
    }
    if (useShell) {
      // /S - strip first and last quotes and execute everything else as is.
      // /E:ON - enable extended command set.
      // /V:ON - enable delayed variable expansion
      // /D - ignore AutoRun registry entries.
      // /C - execute command. This must be the last option before the command itself.
      return new String[] {
        "CMD.EXE", "/S", "/E:ON", "/V:ON", "/D", "/C", Joiner.on(' ').join(modifiedArgv)
      };
    } else {
      return modifiedArgv.toArray(new String[argv.size()]);
    }
  }

  public Command build() {
    Preconditions.checkState(system != OS.UNKNOWN, "Unidentified operating system");
    Preconditions.checkNotNull(workingDir, "Working directory must be set");
    Preconditions.checkState(!argv.isEmpty(), "At least one argument is expected");

    return new Command(
        system == OS.WINDOWS ? transformArgvForWindows() : transformArgvForLinux(),
        env,
        workingDir,
        clientEnv);
  }
}
