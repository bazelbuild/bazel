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
import com.google.devtools.build.lib.view.SourceManifestAction;

/**
 * A context for {@link SourceManifestAction} that uses the runtime to determine
 * the workspace suffix.
 */
@ExecutionStrategy(contextType = SourceManifestAction.Context.class)
public class SourceManifestActionContextImpl implements SourceManifestAction.Context {
  private final String workspaceName;

  public SourceManifestActionContextImpl(String workspaceSuffix) {
    this.workspaceName = workspaceSuffix;
  }

  @Override
  public String getWorkspaceName() {
    return workspaceName;
  }
}
