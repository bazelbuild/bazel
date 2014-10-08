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

package com.google.devtools.build.lib.view.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.syntax.Label;

import java.io.IOException;
import java.io.OutputStream;

/**
 * Abstract Action to write to a file.
 */
public abstract class AbstractFileWriteAction extends AbstractAction {

  protected final boolean makeExecutable;

  /**
   * Creates a new AbstractFileWriteAction instance.
   *
   * @param owner the action owner.
   * @param inputs the Artifacts that this Action depends on
   * @param output the Artifact that will be created by executing this Action.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public AbstractFileWriteAction(ActionOwner owner,
      Iterable<Artifact> inputs, Artifact output, boolean makeExecutable) {
    // There is only one output, and it is primary.
    super(owner, inputs, ImmutableList.of(output));
    this.makeExecutable = makeExecutable;
  }

  public boolean makeExecutable() {
    return makeExecutable;
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    try {
      getStrategy(actionExecutionContext.getExecutor()).exec(actionExecutionContext.getExecutor(),
          this, actionExecutionContext.getFileOutErr());
    } catch (ExecException e) {
      throw e.toActionExecutionException(
          "Writing file for rule '" + Label.print(getOwner().getLabel()) + "'",
          actionExecutionContext.getExecutor().getVerboseFailures(), this);
    }
  }

  /**
   * Write the content of the output file to the provided output stream.
   *
   * @param out the output stream to write the content to.
   * @param eventHandler destination for warning messages.  (Note that errors should
   *        still be indicated by throwing an exception; reporter.error() will
   *        not cause action execution to fail.)
   * @param executor the Executor.
   * @throws IOException if the content cannot be written to the output stream
   */
  public abstract void writeOutputFile(OutputStream out, EventHandler eventHandler,
      Executor executor) throws IOException, InterruptedException, ExecException;

  // We're mainly doing I/O, so estimate very low CPU usage, e.g. 1%. Just a guess.
  private static final ResourceSet DEFAULT_FILEWRITE_ACTION_RESOURCE_SET =
      new ResourceSet(/*memoryMb=*/0.0, /*cpuUsage=*/0.01, /*ioUsage=*/0.2);

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return DEFAULT_FILEWRITE_ACTION_RESOURCE_SET;
  }

  @Override
  public String getMnemonic() {
    return "FileWrite";
  }

  @Override
  protected String getRawProgressMessage() {
    return "Writing " + (makeExecutable ? "script " : "file ")
    + Iterables.getOnlyElement(getOutputs()).prettyPrint();
  }

  @Override
  public final String describeStrategy(Executor executor) {
    return "local";
  }

  private FileWriteActionContext getStrategy(Executor executor) {
    return executor.getContext(FileWriteActionContext.class);
  }
}
