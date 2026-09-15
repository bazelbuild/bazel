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

import static java.util.Map.Entry.comparingByKey;

import com.google.common.collect.Ordering;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Utils for emitting scripts cross-platform. */
public class ScriptUtil {
  private static final ScriptEmitter SCRIPT_EMITTER =
      OS.getCurrent() == OS.WINDOWS ? new WindowsScriptEmitter() : new LinuxScriptEmitter();

  static void emitBeginIsolate(StringBuilder message) {
    SCRIPT_EMITTER.emitBeginIsolate(message);
  }

  static void emitEndIsolate(StringBuilder message) {
    SCRIPT_EMITTER.emitEndIsolate(message);
  }

  /** Emits command to change directories. */
  public static void emitChangeDirectory(StringBuilder message, String cwd) {
    SCRIPT_EMITTER.emitChangeDirectory(message, cwd);
  }

  /** Emits the "env" prefix, setting and unsetting the provided environment variables. */
  public static void emitEnvPrefix(
      StringBuilder message,
      boolean ignoreEnvironment,
      Map<String, String> setEnv,
      @Nullable List<String> unsetEnv) {
    SCRIPT_EMITTER.emitEnvPrefix(message, ignoreEnvironment);

    if (unsetEnv != null) {
      for (String name : Ordering.natural().sortedCopy(unsetEnv)) {
        message.append("  ");
        SCRIPT_EMITTER.emitUnsetEnvVar(message, name);
      }
    }

    Comparator<Map.Entry<String, String>> mapEntryComparator = comparingByKey();
    for (Map.Entry<String, String> entry :
        Ordering.from(mapEntryComparator).sortedCopy(setEnv.entrySet())) {
      message.append("  ");
      SCRIPT_EMITTER.emitSetEnvVar(message, entry.getKey(), entry.getValue());
    }
  }

  /**
   * Formats the command element and adds it to the message.
   *
   * @param isBinary is true if the `commandElement` is the binary to be executed
   */
  public static void emitCommandElement(
      StringBuilder message, String commandElement, boolean isBinary) {
    SCRIPT_EMITTER.emitCommandElement(message, commandElement, isBinary);
  }

  /** Emits the prefix for "exec"-ing a command. */
  public static void emitExec(StringBuilder message) {
    SCRIPT_EMITTER.emitExec(message);
  }

  private interface ScriptEmitter {
    void emitBeginIsolate(StringBuilder message);

    void emitEndIsolate(StringBuilder message);

    void emitChangeDirectory(StringBuilder message, String cwd);

    void emitEnvPrefix(StringBuilder message, boolean ignoreEnvironment);

    void emitSetEnvVar(StringBuilder message, String name, String value);

    void emitUnsetEnvVar(StringBuilder message, String name);

    void emitCommandElement(StringBuilder message, String commandElement, boolean isBinary);

    void emitExec(StringBuilder message);
  }

  private static final class LinuxScriptEmitter implements ScriptEmitter {

    @Override
    public void emitBeginIsolate(StringBuilder message) {
      message.append("(");
    }

    @Override
    public void emitEndIsolate(StringBuilder message) {
      message.append(")");
    }

    @Override
    public void emitChangeDirectory(StringBuilder message, String cwd) {
      message.append("cd ").append(ShellEscaper.escapeString(cwd)).append(" && \\\n  ");
    }

    @Override
    public void emitEnvPrefix(StringBuilder message, boolean ignoreEnvironment) {
      message.append(ignoreEnvironment ? "env - \\\n  " : "env \\\n  ");
    }

    @Override
    public void emitSetEnvVar(StringBuilder message, String name, String value) {
      message
          .append(ShellEscaper.escapeString(name))
          .append('=')
          .append(ShellEscaper.escapeString(value))
          .append(" \\\n  ");
    }

    @Override
    public void emitUnsetEnvVar(StringBuilder message, String name) {
      // Only the short form of --unset is supported on macOS.
      message.append("-u ").append(ShellEscaper.escapeString(name)).append(" \\\n  ");
    }

    @Override
    public void emitCommandElement(StringBuilder message, String commandElement, boolean isBinary) {
      message.append(ShellEscaper.escapeString(commandElement));
    }

    @Override
    public void emitExec(StringBuilder message) {
      message.append("exec ");
    }
  }

  // TODO(bazel-team): (2010) Add proper escaping. We can't use ShellUtils.shellEscape() as it is
  // incompatible with CMD.EXE syntax, but something else might be needed.
  private static final class WindowsScriptEmitter implements ScriptEmitter {

    @Override
    public void emitBeginIsolate(StringBuilder message) {
      // TODO(bazel-team): Implement this.
    }

    @Override
    public void emitEndIsolate(StringBuilder message) {
      // TODO(bazel-team): Implement this.
    }

    @Override
    public void emitChangeDirectory(StringBuilder message, String cwd) {
      message.append("cd ").append("/d ").append(cwd).append("\n");
    }

    @Override
    public void emitEnvPrefix(StringBuilder message, boolean ignoreEnvironment) {}

    @Override
    public void emitSetEnvVar(StringBuilder message, String name, String value) {
      message.append("SET ").append(name).append('=').append(value).append("\n  ");
    }

    @Override
    public void emitUnsetEnvVar(StringBuilder message, String name) {
      message.append("SET ").append(name).append('=').append("\n  ");
    }

    @Override
    public void emitCommandElement(StringBuilder message, String commandElement, boolean isBinary) {
      // Replace the forward slashes with back slashes if the `commandElement` is the binary path
      message.append(isBinary ? commandElement.replace('/', '\\') : commandElement);
    }

    @Override
    public void emitExec(StringBuilder message) {
      // TODO(bazel-team): Implement this if possible for greater efficiency.
    }
  }

  private ScriptUtil() {}
}
