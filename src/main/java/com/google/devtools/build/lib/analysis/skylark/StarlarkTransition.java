// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.List;
import java.util.Objects;

/** A marker class for configuration transitions that are defined in Starlark. */
public abstract class StarlarkTransition implements ConfigurationTransition {

  private final StarlarkDefinedConfigTransition starlarkDefinedConfigTransition;

  public StarlarkTransition(StarlarkDefinedConfigTransition starlarkDefinedConfigTransition) {
    this.starlarkDefinedConfigTransition = starlarkDefinedConfigTransition;
  }

  public void replayOn(ExtendedEventHandler eventHandler) {
    starlarkDefinedConfigTransition.getEventHandler().replayOn(eventHandler);
    starlarkDefinedConfigTransition.getEventHandler().clear();
  }

  public boolean hasErrors() {
    return starlarkDefinedConfigTransition.getEventHandler().hasErrors();
  }

  /** Exception class for exceptions thrown during application of a starlark-defined transition */
  public static class TransitionException extends Exception {
    public TransitionException(String message) {
      super(message);
    }
  }

  /**
   * Method to be called after Starlark-transitions are applied. Logs any events (e.g. {@code
   * print()}s, errors} to output and throws an error if we had any errors. Right now, Starlark
   * transitions are the only kind that knows how to throw errors so we know this will only report
   * and throw if a Starlark transition caused a problem.
   *
   * @param eventHandler eventHandler to replay events on
   * @param root transition that was applied. Could be a composing transition so we pull and
   *     post-process all StarlarkTransitions out of whatever transition is passed here
   * @throws TransitionException if an error occurred during Starlark transition application.
   */
  public static void postProcessStarlarkTransitions(
      ExtendedEventHandler eventHandler, ConfigurationTransition root) throws TransitionException {
    List<ConfigurationTransition> transitions = ComposingTransition.decomposeTransition(root);
    for (ConfigurationTransition transition : transitions) {
      if (transition instanceof StarlarkTransition) {
        StarlarkTransition starlarkTransition = (StarlarkTransition) transition;
        boolean hasErrors = starlarkTransition.hasErrors();
        starlarkTransition.replayOn(eventHandler);
        if (hasErrors) {
          throw new TransitionException("Errors encountered while applying Starlark transition");
        }
      }
    }
  }

  @Override
  public boolean equals(Object object) {
    if (object == this) {
      return true;
    }
    if (object instanceof StarlarkTransition) {
      StarlarkDefinedConfigTransition starlarkDefinedConfigTransition =
          ((StarlarkTransition) object).starlarkDefinedConfigTransition;
      return Objects.equals(starlarkDefinedConfigTransition, this.starlarkDefinedConfigTransition);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(starlarkDefinedConfigTransition);
  }
}
