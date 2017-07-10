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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ExecException;
import java.io.IOException;

/**
 * Context for actions that do include scanning.
 */
public interface IncludeScanningContext extends ActionContext {
  /**
   * Extracts the set of include files from a source file.
   *
   * @param actionExecutionContext the execution context
   * @param resourceOwner the resource owner
   * @param primaryInput the source file to be include scanned
   * @param primaryOutput the output file where the results should be put
   */
  void extractIncludes(
      ActionExecutionContext actionExecutionContext,
      Action resourceOwner,
      Artifact primaryInput,
      Artifact primaryOutput)
      throws IOException, ExecException, InterruptedException;

  /**
   * Returns the artifact resolver.
   */
  ArtifactResolver getArtifactResolver();
}
