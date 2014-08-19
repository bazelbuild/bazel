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
package com.google.devtools.build.lib.actions;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsClassProvider;

/**
 * The Executor provides the context for the execution of actions. It is only valid during the
 * execution phase, and references should not be cached.
 *
 * <p>This class provides the actual logic to execute actions. The platonic ideal of this system
 * is that {@link Action}s are immutable objects that tell Blaze <b>what</b> to do and
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
public interface Executor {
  /**
   * A marker interface for classes that provide services for actions during execution.
   *
   * <p>Interfaces extending this one should also be annotated with {@link ActionContextMarker}.
   */
  public interface ActionContext {
  }

  /**
   * Returns the execution root. This is the directory underneath which Blaze builds its entire
   * output working tree, including the source symlink forest. All build actions are executed
   * relative to this directory.
   */
  Path getExecRoot();

  /**
   * Returns the action graph.
   *
   * <p>Using this is non-hermetic, so please only do it with good reason. The only current use is
   * handling C++ includes.
   */
  ActionGraph getActionGraph();

  /**
   * Returns a clock. This is not hermetic, and should only be used for build info actions or
   * performance measurements / reporting.
   */
  Clock getClock();

  /**
   * The EventBus for the current build.
   */
  EventBus getEventBus();

  /**
   * Returns whether failures should have verbose error messages.
   */
  boolean getVerboseFailures();

  /**
   * Returns the command line options of the Blaze command being executed.
   */
  OptionsClassProvider getOptions();

  /**
   * Whether this Executor reports subcommands. If not, reportSubcommand has no effect.
   * This is provided so the caller of reportSubcommand can avoid wastefully constructing the
   * subcommand string.
   */
  boolean reportsSubcommands();

  /**
   * Report a subcommand event to this Executor's Reporter and, if action
   * logging is enabled, post it on its EventBus.
   */
  void reportSubcommand(String reason, String message);

  /**
   * An event listener to report warnings to; actions should not use this to report errors, instead
   * they should throw an ActionExecutionException. Calling showOutput on the returned object is
   * not hermetic.
   */
  // TODO(bazel-team): This should be removed in favor of reporting problems / debug output to
  // OutErr.
  EventHandler getReporter();

  /**
   * Looks up and returns an action context implementation of the given interface type.
   */
  <T extends ActionContext> T getContext(Class<? extends T> type);

  /**
   * Returns the action context implementation for spawn actions with a given mnemonic.
   */
  SpawnActionContext getSpawnActionContext(String mnemonic);
}
