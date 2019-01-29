// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.standalone;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.io.IOException;

/**
 * An IncludeExtractionContext that does nothing. Since local execution does not need to discover
 * inclusion in advance, we do not need include scanning.
 */
@ExecutionStrategy(contextType = CppIncludeExtractionContext.class)
class DummyCppIncludeExtractionContext implements CppIncludeExtractionContext {
  private final CommandEnvironment env;

  public DummyCppIncludeExtractionContext(CommandEnvironment env) {
    this.env = env;
  }

  @Override
  public void extractIncludes(
      ActionExecutionContext actionExecutionContext,
      Action resourceOwner,
      Artifact primaryInput,
      Artifact primaryOutput,
      Artifact grepIncludes)
      throws IOException {
    FileSystemUtils.writeContent(
        actionExecutionContext.getInputPath(primaryOutput), new byte[] {});
  }

  @Override
  public ArtifactResolver getArtifactResolver() {
    return env.getSkyframeBuildView().getArtifactFactory();
  }
}
