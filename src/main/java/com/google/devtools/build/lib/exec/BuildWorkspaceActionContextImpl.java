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
package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.blaze.VerifiableOption;
import com.google.devtools.build.lib.view.BuildWorkspaceActionContext;
import com.google.devtools.common.options.OptionsProvider;

/**
 * Action Context for Build Workspace actions.
 */
@ExecutionStrategy(contextType = BuildWorkspaceActionContext.class)
public final class BuildWorkspaceActionContextImpl implements BuildWorkspaceActionContext {
  private final OptionsProvider options;

  public BuildWorkspaceActionContextImpl(OptionsProvider options) {
    this.options = options;
  }

  @Override
  public boolean isVerifiable() {
    return options.getOptions(VerifiableOption.class).verifiable;
  }
}
