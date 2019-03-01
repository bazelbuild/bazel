// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.graph.ImmutableGraph;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Interface of an Action that is Skyframe-aware.
 *
 * <p><b>IMPORTANT</b>: actions that implement this interface should override
 * {@code Action.executeUnconditionally} and return true. See below for details.
 *
 * <p>Implementors of this interface can request Skyframe dependencies to perform arbitrary
 * computation or establish desired dependencies before they are executed but after their inputs
 * have been built.
 *
 * <p>The {@link ActionExecutionFunction} will make sure that all requested SkyValues are built and
 * that the {@link SkyframeAwareAction#establishSkyframeDependencies(Environment)} function
 * completed successfully before attempting to execute the action.
 *
 * <p><b>It is crucial to correct action reexecution that implementors override
 * {@code Action.executeUnconditionally} to always return true.</b> Skyframe tracks changes in both
 * the input files and in dependencies established through
 * {@link #establishSkyframeDependencies(Environment)}, but the action cache only knows about the
 * input files. So if only the extra "skyframe dependencies" change, the action cache will believe
 * the action to be up-to-date and skip actual execution. Therefore it's crucial to bypass action
 * cache checking by marking the action as unconditionally executed.
 */
public interface SkyframeAwareAction {

  /** Wrapper and/or base class for exceptions raised in {@link #establishSkyframeDependencies}. */
  class ExceptionBase extends Exception {
    public ExceptionBase(String message) {
      super(message);
    }

    public ExceptionBase(Throwable cause) {
      super(cause.getMessage(), cause);
    }
  }

  /**
   * Establish dependencies on Skyframe values before executing the action.
   *
   * <p><b>IMPORTANT</b>: actions that implement this interface should override {@code
   * Action.executeUnconditionally} and return true. See {@link SkyframeAwareAction} why.
   *
   * <p>This method should perform as little computation as possible: ideally it should request one
   * or a few SkyValues, perhaps set some state somewhere and return. If this method needs to
   * perform anything more complicated than that, including perhaps some non-trivial computation,
   * you should implement that as a SkyFunction and request the corresponding SkyValue in this
   * method.
   *
   * <p>Because the requested SkyValues may not yet be present in the graph, this method must be
   * safe to call multiple times, and should always leave the action object in a consistent state.
   *
   * <p>This method should not attempt to handle errors or missing dependencies (other than wrapping
   * exceptions); that is the responsibility of the caller. It should return as soon as possible,
   * ready to be called again at a later time if need be.
   *
   * <p>The return value will be incorporated into the
   * {@link com.google.devtools.build.lib.actions.ActionExecutionContext}.
   */
  Object establishSkyframeDependencies(Environment env) throws ExceptionBase, InterruptedException;

  /**
   * Returns the Skyframe nodes which need to be rewound if a consumer of this action's output finds
   * out that output has been lost.
   */
  ImmutableGraph<SkyKey> getSkyframeDependenciesForRewinding(SkyKey self);
}
