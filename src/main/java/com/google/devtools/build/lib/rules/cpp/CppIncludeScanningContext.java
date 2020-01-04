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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Context for include scanning.
 */
@ActionContextMarker(name = "IncludeScanning")
public interface CppIncludeScanningContext extends ActionContext {
  /**
   * Does include scanning to find the list of files needed to execute the action.
   *
   * <p>Returns null if additional inputs will only be found during action execution, not before.
   */
  @Nullable
  ListenableFuture<List<Artifact>> findAdditionalInputs(
      CppCompileAction action,
      ActionExecutionContext actionExecutionContext,
      IncludeProcessing includeProcessing,
      IncludeScanningHeaderData includeScanningHeaderData)
      throws ExecException, InterruptedException, ActionExecutionException;
}
