// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionContext.ActionContextRegistry;
import javax.annotation.Nullable;

/** Restricts action-context lookup to the capabilities declared by a process-free action. */
public final class ProcessFreeActionContextRegistry implements ActionContextRegistry {
  private final ActionAnalysisMetadata action;
  private final ActionContextRegistry delegate;
  private final ImmutableSet<Class<? extends ActionContext>> allowedContexts;

  public ProcessFreeActionContextRegistry(
      ActionAnalysisMetadata action, ActionContextRegistry delegate) {
    this.action = checkNotNull(action);
    this.delegate = checkNotNull(delegate);
    checkArgument(action.isProcessFree(), "action is not process-free: %s", action.prettyPrint());
    this.allowedContexts = action.getProcessFreeActionContexts();
  }

  @Override
  @Nullable
  public <T extends ActionContext> T getContext(Class<T> identifyingType) {
    checkState(
        allowedContexts.contains(identifyingType),
        "process-free action %s requested undeclared execution context %s",
        action.prettyPrint(),
        identifyingType.getName());
    return delegate.getContext(identifyingType);
  }
}
