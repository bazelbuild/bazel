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

import com.google.devtools.build.lib.actions.ActionExecutionContext.ShowSubcommands;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsProvider;

/**
 * The Executor provides the context for the execution of actions. It is only valid during the
 * execution phase, and references should not be cached.
 *
 * <p>This class provides the actual logic to execute actions. The platonic ideal of this system is
 * that {@link Action}s are immutable objects that tell Blaze <b>what</b> to do and
 * <link>ActionContext</link>s tell Blaze <b>how</b> to do it (however, we do have an "execute"
 * method on actions now).
 *
 * <p>In theory, most of the methods below would not exist and they would be methods on action
 * contexts, but in practice, that would require some refactoring work so we are stuck with these
 * for the time being.
 *
 * <p>In theory, we could also merge {@link Executor} with {@link ActionExecutionContext}, since
 * they both provide services to actions being executed and are passed to almost the same places.
 */
public interface Executor extends ActionContext.ActionContextRegistry {

  /** Returns the file system of blaze. */
  FileSystem getFileSystem();

  /**
   * Returns the execution root. This is the directory underneath which Blaze builds its entire
   * output working tree, including the source symlink forest. All build actions are executed
   * relative to this directory.
   */
  Path getExecRoot();

  /**
   * Returns a clock. This is not hermetic, and should only be used for build info actions or
   * performance measurements / reporting.
   */
  Clock getClock();

  /**
   * Returns whether failures should have verbose error messages.
   */
  boolean getVerboseFailures();

  /**
   * Returns the command line options of the Blaze command being executed.
   */
  OptionsProvider getOptions();

  /**
   * Whether this Executor reports subcommands. If not, reportSubcommand has no effect.
   * This is provided so the caller of reportSubcommand can avoid wastefully constructing the
   * subcommand string.
   */
  ShowSubcommands reportsSubcommands();
}
