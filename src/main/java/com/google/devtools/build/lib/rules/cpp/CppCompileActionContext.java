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

import com.google.devtools.build.lib.actions.ActionContextMarker;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.actions.ResourceSet;

import java.io.IOException;
import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Context for compiling plain C++.
 */
@ActionContextMarker(name = "C++")
public interface CppCompileActionContext extends ActionContext {
  /**
   * Reply for the execution of a C++ compilation.
   */
  public interface Reply {
    /**
     * Returns the contents of the .d file.
     */
    byte[] getContents() throws IOException;
  }

  /**
   * Does include scanning to find the list of files needed to execute the action.
   *
   * <p>Returns null if additional inputs will only be found during action execution, not before.
   * </p>
   */
  @Nullable
  public Collection<Artifact> findAdditionalInputs(CppCompileAction action,
      ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException, ActionExecutionException;

  /**
   * Executes the given action and return the reply of the executor.
   */
  Reply execWithReply(CppCompileAction action,
      ActionExecutionContext actionExecutionContext) throws ExecException, InterruptedException;

  /**
   * Returns the executor reply from an exec exception, if available.
   */
  @Nullable Reply getReplyFromException(
      ExecException e, CppCompileAction action);

  /**
   * Returns the estimated resource consumption of the action.
   */
  ResourceSet estimateResourceConsumption(CppCompileAction action);

  /**
   * Returns whether include scanning needs to be run.
   */
  boolean needsIncludeScanning();

  /**
   * Returns the include files that should be shipped to the executor in addition the ones that
   * were declared.
   */
  Collection<Artifact> getScannedIncludeFiles(
      CppCompileAction action, ActionExecutionContext actionExecutionContext)
          throws ActionExecutionException, InterruptedException;
}
