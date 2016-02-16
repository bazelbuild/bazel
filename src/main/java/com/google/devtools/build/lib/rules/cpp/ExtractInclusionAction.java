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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;

import java.io.IOException;

/**
 * An action which greps for includes over a given .cc or .h file.
 * This is a part of the work required for C++ include scanning.
 *
 * <p>For generated files, it is advantageous to do this remotely, to avoid having to download
 * the generated file.
 *
 * <p>Note that this may run grep-includes over-optimistically, where we previously
 * had not. For example, consider a cc_library of generated headers. If another
 * library depends on it, and only references one of the headers, the other
 * grep-includes will have been wasted.
 */
final class ExtractInclusionAction extends AbstractAction {

  private static final String GUID = "45b43e5a-4734-43bb-a05e-012313808142";

  /**
   * Constructs a new action.
   */
  public ExtractInclusionAction(ActionOwner owner, Artifact input, Artifact output) {
    super(owner, ImmutableList.of(input), ImmutableList.of(output));
  }

  @Override
  protected String computeKey() {
    return GUID;
  }

  @Override
  public String getMnemonic() {
    return "GrepIncludes";
  }

  @Override
  protected String getRawProgressMessage() {
    return "Extracting include lines from " + getPrimaryInput().prettyPrint();
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return ResourceSet.ZERO;
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    IncludeScanningContext context = executor.getContext(IncludeScanningContext.class);
    try {
      context.extractIncludes(actionExecutionContext, this, getPrimaryInput(),
          getPrimaryOutput());
    } catch (IOException e) {
      throw new ActionExecutionException(e, this, false);
    } catch (ExecException e) {
      throw e.toActionExecutionException(this);
    }
  }
}
