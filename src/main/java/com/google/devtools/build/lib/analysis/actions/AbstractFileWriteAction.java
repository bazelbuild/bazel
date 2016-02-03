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

package com.google.devtools.build.lib.analysis.actions;

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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;

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
  public final void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    try {
      getStrategy(actionExecutionContext.getExecutor()).exec(actionExecutionContext.getExecutor(),
          this, actionExecutionContext.getFileOutErr(), actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException(
          "Writing file for rule '" + Label.print(getOwner().getLabel()) + "'",
          actionExecutionContext.getExecutor().getVerboseFailures(), this);
    }
    afterWrite(actionExecutionContext.getExecutor());
  }

  /**
   * Produce a DeterministicWriter that can write the file to an OutputStream deterministically.
   *
   * @param eventHandler destination for warning messages.  (Note that errors should
   *        still be indicated by throwing an exception; reporter.error() will
   *        not cause action execution to fail.)
   * @param executor the Executor.
   * @throws IOException if the content cannot be written to the output stream
   */
  public abstract DeterministicWriter newDeterministicWriter(EventHandler eventHandler,
      Executor executor) throws IOException, InterruptedException, ExecException;

  /**
   * This hook is called after the File has been successfully written to disk.
   *
   * @param executor the Executor.
   */
  protected void afterWrite(Executor executor) {
  }

  // We're mainly doing I/O, so estimate very low CPU usage, e.g. 1%. Just a guess.
  private static final ResourceSet DEFAULT_FILEWRITE_LOCAL_ACTION_RESOURCE_SET =
      ResourceSet.createWithRamCpuIo(/*memoryMb=*/0.0, /*cpuUsage=*/0.01, /*ioUsage=*/0.2);

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return executor.getContext(FileWriteActionContext.class).estimateResourceConsumption(this);
  }

  public ResourceSet estimateResourceConsumptionLocal() {
    return DEFAULT_FILEWRITE_LOCAL_ACTION_RESOURCE_SET;
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

  /**
   * Whether the file write can be generated remotely. If the file is consumed in Blaze
   * unconditionally, it doesn't make sense to run remotely.
   */
  public boolean isRemotable() {
    return true;
  }

  private FileWriteActionContext getStrategy(Executor executor) {
    return executor.getContext(FileWriteActionContext.class);
  }

  /**
   * A deterministic writer writes bytes to an output stream. The same byte stream is written
   * on every invocation of writeOutputFile().
   */
  public interface DeterministicWriter {
    public void writeOutputFile(OutputStream out) throws IOException;
  }
}
