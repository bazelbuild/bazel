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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ActionContext;

/** An {@link ActionContextProvider} that just provides the {@link ActionContext}s it's given. */
public final class SimpleActionContextProvider<T extends ActionContext, C extends T>
    extends ActionContextProvider {
  private final C context;
  private final Class<T> identifyingType;
  private final String[] commandLineIdentifiers;

  /**
   * Creates a provider which will register the given context with the passed identifying type and
   * commandline identifiers.
   *
   * @see ActionContextCollector
   */
  public SimpleActionContextProvider(
      Class<T> identifyingType, C context, String... commandLineIdentifiers) {
    this.context = context;
    this.identifyingType = identifyingType;
    this.commandLineIdentifiers = commandLineIdentifiers;
  }

  @Override
  public void registerActionContexts(ActionContextCollector collector) {
    collector.forType(identifyingType).registerContext(context, commandLineIdentifiers);
  }
}
