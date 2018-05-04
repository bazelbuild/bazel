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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.rules.fileset.FilesetActionContext;

/**
 * Context for Fileset manifest actions. It currently only provides a ThreadPoolExecutor.
 *
 * <p>Fileset is a legacy, google-internal mechanism to make parts of the source tree appear as a
 * tree in the output directory.
 */
@ExecutionStrategy(contextType = FilesetActionContext.class)
public final class FilesetActionContextImpl implements FilesetActionContext {
  // TODO(bazel-team): it would be nice if this weren't shipped in Bazel at all.

  /**
   * Factory class.
   */
  public static class Provider extends ActionContextProvider {
    private FilesetActionContextImpl impl;

    public Provider(String workspaceName) {
      this.impl = new FilesetActionContextImpl(workspaceName);
    }

    @Override
    public Iterable<? extends ActionContext> getActionContexts() {
      return ImmutableList.of(impl);
    }
  }

  private final String workspaceName;

  private FilesetActionContextImpl(String workspaceName) {
    this.workspaceName = workspaceName;
  }

  @Override
  public String getWorkspaceName() {
    return workspaceName;
  }
}
