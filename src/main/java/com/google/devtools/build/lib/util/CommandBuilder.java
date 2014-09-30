// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.CharMatcher;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Implements OS aware {@link Command} builder. At this point only Linux and
 * Windows XP are supported.
 *
 * <p>Builder will also apply heuristic to identify trivial cases where
 * unix-like command lines could be automatically converted into the
 * Windows-compatible form.
 *
 * <p>TODO(bazel-team): (2010) Some of the code here is very similar to the
 * {@link com.google.devtools.lib.shell.Shell} class. This should be looked at.
 */
public final class CommandBuilder {

  private static final String WINDOWS_PYTHON_PROPERTY_NOT_SET =
      "blaze.windows.python.dir property has not been set and is required to run Python " +
      "scripts on Windows.";

  private static final List<String> SHELLS = ImmutableList.of("/bin/sh", "/bin/bash");

  // The operating system we are running on.
  // TODO(bazel-team): should Darwin & Linux be merged into unix?
  public enum OS { DARWIN, LINUX, WINDOWS, UNKNOWN }

  private static final OS HOST_SYSTEM =
      "Mac OS X".equals(System.getProperty("os.name")) ? OS.DARWIN : (
      "Linux".equals(System.getProperty("os.name")) ? OS.LINUX : (
      "Windows XP".equals(System.getProperty("os.name")) ? OS.WINDOWS : (
      "Windows 7".equals(System.getProperty("os.name")) ? OS.WINDOWS : OS.UNKNOWN)));

  // On Windows, there is no standard Python interpreter, so we will need to
  // find one - e.g. the one bundled with g4. Blaze startup script would locate it
  // and set property below to the location of the python installation directory.
  private static final String DEFAULT_WINDOWS_PYTHON_DIR =
      System.getProperty("blaze.windows.python.dir");

  private static final String DEFAULT_LINUX_PYTHON_DIR =
      "/usr/grte/v3/k8-linux/bin/python2.7";
  private static final String DEFAULT_DARWIN_PYTHON_DIR =
      "/usr/bin/python2.6";

  private static final Splitter ARGV_SPLITTER = Splitter.on(CharMatcher.anyOf(" \t"));

  private final OS system;
  private final List<String> argv = new ArrayList<>();
  private final Map<String, String> env = new HashMap<>();
  private File workingDir = null;
  private boolean useShell = false;
  private String windowsPythonExePath;

  static OS getHostSystem() {
    return HOST_SYSTEM;
  }

  public CommandBuilder() {
    this(HOST_SYSTEM);
  }

  @VisibleForTesting
  CommandBuilder(OS system) {
    this.system = system;
    // It is very important not to set windowsPythonExePath to just "python.exe". If only
    // basename without path is used, Windows can use registry entry under
    // "HKLM\Software\Microsoft\Windows\CurrentVersion\App Paths" (if present) to select
    // executable from the predefined path - which would work even with sanitized PATH.
    // So we ensure that Blaze receives location of the python installation directory when
    // running on Windows.
    this.windowsPythonExePath = Strings.isNullOrEmpty(DEFAULT_WINDOWS_PYTHON_DIR)
        ? null // disable Python support on Windows.
        : DEFAULT_WINDOWS_PYTHON_DIR + PathFragment.SEPARATOR_CHAR + "python.exe";
  }

  public CommandBuilder addArg(String arg) {
    Preconditions.checkNotNull(arg, "Argument must not be null");
    argv.add(arg);
    return this;
  }

  public CommandBuilder addArgs(Iterable<String> args) {
    Preconditions.checkArgument(!Iterables.contains(args, null), "Arguments must not be null");
    Iterables.addAll(argv, args);
    return this;
  }

  public CommandBuilder addArgs(String... args) {
    return addArgs(Arrays.asList(args));
  }

  public CommandBuilder addPythonExecutable() {
    this.env.put("PYTHONHASHSEED", "0");
    return addArgs(getPythonExecutable());
  }

  public CommandBuilder addEnv(Map<String, String> env) {
    Preconditions.checkNotNull(env);
    this.env.putAll(env);
    return this;
  }

  public CommandBuilder emptyEnv() {
    env.clear();
    return this;
  }

  public CommandBuilder setEnv(Map<String, String> env) {
    emptyEnv();
    addEnv(env);
    return this;
  }

  public CommandBuilder setWorkingDir(Path path) {
    Preconditions.checkNotNull(path);
    workingDir = path.getPathFile();
    return this;
  }

  public CommandBuilder useTempDir() {
    workingDir = new File(System.getProperty("java.io.tmpdir"));
    return this;
  }

  public CommandBuilder useShell(boolean useShell) {
    this.useShell = useShell;
    return this;
  }

  @VisibleForTesting
  CommandBuilder setWindowsPythonExecutable(String pythonInterpreter) {
    Preconditions.checkState(system == OS.WINDOWS, "This method can be used only on Windows");
    Preconditions.checkNotNull(pythonInterpreter);
    Preconditions.checkState(!"python.exe".equalsIgnoreCase(pythonInterpreter),
        "Python interpreter must not be set to 'python.exe' to prevent Windows to select it's "
        + "location using registry entry");
    this.windowsPythonExePath = pythonInterpreter;
    return this;
  }

  private String getPythonExecutable() {
    switch (HOST_SYSTEM) {
      case WINDOWS:
        Preconditions.checkNotNull(windowsPythonExePath, WINDOWS_PYTHON_PROPERTY_NOT_SET);
        return windowsPythonExePath;
      case LINUX:
        return DEFAULT_LINUX_PYTHON_DIR;
      case DARWIN:
        return DEFAULT_DARWIN_PYTHON_DIR;
      default:
        Preconditions.checkState(false, "Unknown OS %s", HOST_SYSTEM);
        return null; // To keep the compiler happy - won't ever get here.
    }
  }

  private boolean argvStartsWithSh() {
    return argv.size() >= 2 && SHELLS.contains(argv.get(0)) && "-c".equals(argv.get(1));
  }

  private String[] transformArgvForLinux() {
    // If command line already starts with "/bin/sh -c", ignore useShell attribute.
    if (useShell && !argvStartsWithSh()) {
      // c.g.io.base.shell.Shell.shellify() actually concatenates argv into the space-separated
      // string here. Not sure why, but we will do the same.
      return new String[] { "/bin/sh", "-c", Joiner.on(' ').join(argv) };
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
      // For python programs, prepend python executable.
      if (command.endsWith(".py") || command.endsWith(".pyc")) {
        Preconditions.checkNotNull(windowsPythonExePath, WINDOWS_PYTHON_PROPERTY_NOT_SET);
        modifiedArgv.add(0, windowsPythonExePath);
        command = windowsPythonExePath;
      }
      // Automatically enable CMD.EXE use if we are executing something else besides "*.exe" file.
      if (!command.toLowerCase().endsWith(".exe")) {
        useShell = true;
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
      return new String[] { "CMD.EXE", "/S", "/E:ON", "/V:ON", "/D", "/C",
          "\"" + Joiner.on(' ').join(modifiedArgv) + "\"" };
    } else {
      return modifiedArgv.toArray(new String[argv.size()]);
    }
  }

  public Command build() {
    Preconditions.checkState(system != OS.UNKNOWN, "Unidentified operating system");
    Preconditions.checkNotNull(workingDir, "Working directory must be set");
    Preconditions.checkState(argv.size() > 0, "At least one argument is expected");

    return new Command(
        system == OS.WINDOWS ? transformArgvForWindows() : transformArgvForLinux(),
        env, workingDir);
  }
}
