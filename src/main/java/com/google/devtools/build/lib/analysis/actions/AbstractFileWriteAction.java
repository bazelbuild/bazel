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
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collections;

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
  public final ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Actions.prefetchInputs(getInputs(), actionExecutionContext, this);
    ActionResult actionResult;
    try {
      DeterministicWriter deterministicWriter;
      try {
        deterministicWriter = newDeterministicWriter(actionExecutionContext);
      } catch (IOException e) {
        // Message is a bit misleading but is good enough for the end user.
        throw new EnvironmentalExecException("failed to create file '"
            + getPrimaryOutput().prettyPrint() + "' due to I/O error: " + e.getMessage(), e);
      }
      actionResult =
          ActionResult.create(
              getStrategy(actionExecutionContext)
                  .writeOutputToFile(
                      this,
                      actionExecutionContext,
                      deterministicWriter,
                      makeExecutable,
                      isRemotable()));
    } catch (ExecException e) {
      throw e.toActionExecutionException(
          "Writing file for rule '" + Label.print(getOwner().getLabel()) + "'",
          actionExecutionContext.getVerboseFailures(),
          this);
    }
    afterWrite(actionExecutionContext);
    return actionResult;
  }

  /**
   * Produce a DeterministicWriter that can write the file to an OutputStream deterministically.
   *
   * @param ctx context for use with creating the writer.  
   */
  public abstract DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws IOException, InterruptedException, ExecException;
  
  /**
   * This hook is called after the File has been successfully written to disk.
   *
   * @param actionExecutionContext the execution context
   */
  protected void afterWrite(ActionExecutionContext actionExecutionContext) {
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

  private FileWriteActionContext getStrategy(ActionExecutionContext actionExecutionContext) {
    return actionExecutionContext.getContext(FileWriteActionContext.class);
  }

  /**
   * A deterministic writer writes bytes to an output stream. The same byte stream is written
   * on every invocation of writeOutputFile().
   */
  public interface DeterministicWriter {
    void writeOutputFile(OutputStream out) throws IOException;

    /**
     * Returns the contents that would be written, as a {@link ByteString}. Used when the caller
     * wants a {@link ByteString} in the end, to avoid making unnecessary copies.
     */
    default ByteString getBytes() throws IOException {
      ByteString.Output out = ByteString.newOutput();
      writeOutputFile(out);
      return out.toByteString();
    }
  }
}
