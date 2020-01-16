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
package com.google.devtools.build.lib.exec;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.analysis.ArtifactsToOwnerLabels;

/**
 * An object that provides execution strategies to {@link BlazeExecutor}.
 *
 * <p>For more information, see {@link ExecutorBuilder}.
 */
public abstract class ActionContextProvider {

  /** Called when the executor is constructed. */
  public void executorCreated() throws ExecutorInitException {}

  /** Called when the execution phase is started. */
  public void executionPhaseStarting(
      ActionGraph actionGraph, Supplier<ArtifactsToOwnerLabels> topLevelArtifactsToOwnerLabels)
      throws ExecutorInitException, InterruptedException {}

  /**
   * Called when the execution phase is finished.
   */
  public void executionPhaseEnding() {}

  /** Registers any action contexts originating with this provider with the given collector. */
  public abstract void registerActionContexts(ActionContextCollector collector);

  /**
   * Collector of action contexts. Usage:
   *
   * <p>
   *
   * <pre>{@code
   * collector.forType(MyActionContextType.class)
   *     .registerContext(new MyActionContext(), "commandlineIdentifier")
   *     .registerContext(new MyOtherContext(), "otherIdentifier", "andYetAnother");
   * }</pre>
   *
   * <p>Note that action contexts registered later in time will take precedence over those
   * registered earlier if they share an identifying {@linkplain #forType type} and {@linkplain
   * TypeCollector#registerContext commandline identifier}.
   */
  public interface ActionContextCollector {

    /**
     * Returns a context collector for the given identifying type (typically an action
     * context-extending interface).
     */
    <T extends ActionContext> TypeCollector<T> forType(Class<T> type);

    /** Collector for action contexts of a particular identifying type. */
    interface TypeCollector<T extends ActionContext> {

      /**
       * Registers a context instance implementing this context's type which can optionally be
       * distinguished from other instances of the same identifying type by the given commandline
       * identifiers.
       */
      TypeCollector<T> registerContext(T context, String... commandlineIdentifiers);
    }
  }
}
