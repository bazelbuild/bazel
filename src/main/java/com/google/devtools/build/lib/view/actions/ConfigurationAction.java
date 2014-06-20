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

package com.google.devtools.build.lib.view.actions;

import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

/**
 * A ConfigurationAction is just an AbstractAction with a BuildConfiguration.
 *
 * This is a separate class because we don't want the AbstractAction base class
 * to depend on lib.view.*.
 */
public abstract class ConfigurationAction extends AbstractAction {

  protected final BuildConfiguration configuration;

  /**
   * Constructs a ConfigurationAction.
   *
   * @param owner the action owner.
   * @param inputs the set of all files potentially read by this action; must
   * not be subsequently modified.
   * @param outputs the set of all files written by this action; must not be
   * subsequently modified.
   * @param configuration the BuildConfiguration used to setup this action
   */
  protected ConfigurationAction(ActionOwner owner, Iterable<Artifact> inputs,
      Iterable<Artifact> outputs, BuildConfiguration configuration) {
    super(owner, inputs, outputs);
    this.configuration = configuration;
  }

  /**
   * Returns the BuildConfiguration used for this build.
   */
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  protected ActionExecutionException newActionExecutionException(String messagePrefix,
                                                                 ExecException e,
                                                                 boolean verboseFailures) {
    return e.toActionExecutionException(messagePrefix, verboseFailures, this);
  }
}

