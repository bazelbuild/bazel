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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/** Action to write a parameter file for a {@link CommandLine}. */
@Immutable // if commandLine is immutable
@AutoCodec
public final class ParameterFileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "45f678d8-e395-401e-8446-e795ccc6361f";

  private final CommandLine commandLine;
  private final ParameterFileType type;
  private final boolean hasInputArtifactToExpand;

  /**
   * Creates a new instance.
   *
   * @param owner the action owner
   * @param output the Artifact that will be created by executing this Action
   * @param commandLine the contents to be written to the file
   * @param type the type of the file
   */
  public ParameterFileWriteAction(
      ActionOwner owner, Artifact output, CommandLine commandLine, ParameterFileType type) {
    this(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, commandLine, type);
  }

  /**
   * Creates a new instance.
   *
   * @param owner the action owner
   * @param inputs the list of TreeArtifacts that must be resolved and expanded before evaluating
   *     the contents of {@link CommandLine}.
   * @param output the Artifact that will be created by executing this Action
   * @param commandLine the contents to be written to the file
   * @param type the type of the file
   */
  @AutoCodec.Instantiator
  public ParameterFileWriteAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Artifact output,
      CommandLine commandLine,
      ParameterFileType type) {
    super(owner, inputs, output, false);
    this.commandLine = commandLine;
    this.type = type;
    this.hasInputArtifactToExpand = !inputs.isEmpty();
  }

  @VisibleForTesting
  public CommandLine getCommandLine() {
    return commandLine;
  }

  /**
   * Returns the list of options written to the parameter file. Don't use this method outside tests
   * - the list is often huge, resulting in significant garbage collection overhead.
   *
   * <p>2019-01-10, @leba: Using this method for aquery since it's not performance-critical and the
   * includeParamFile option is flag-guarded with warning regarding output size to user.
   */
  public Iterable<String> getArguments() throws CommandLineExpansionException {
    Preconditions.checkState(
        !hasInputArtifactToExpand,
        "This action contains a CommandLine with TreeArtifacts: %s, which must be expanded using "
        + "ArtifactExpander first before we can evaluate the CommandLine.",
        getInputs());
    return commandLine.arguments();
  }

  @VisibleForTesting
  public String getStringContents() throws CommandLineExpansionException, IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ParameterFile.writeParameterFile(out, getArguments(), type, ISO_8859_1);
    return new String(out.toByteArray(), ISO_8859_1);
  }

  @Override
  public String getStarlarkContent() throws IOException, EvalException {
    if (hasInputArtifactToExpand) {
      // Tree artifact information isn't available at analysis time.
      return null;
    }
    try {
      return getStringContents();
    } catch (CommandLineExpansionException e) {
      throw new EvalException("Error expanding command line: " + e.getMessage());
    }
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws ExecException {
    final Iterable<String> arguments;
    try {
      ArtifactExpander artifactExpander = Preconditions.checkNotNull(ctx.getArtifactExpander());
      arguments = commandLine.arguments(artifactExpander);
    } catch (CommandLineExpansionException e) {
      throw new UserExecException(e);
    }
    return new ParamFileWriter(arguments, type);
  }

  @VisibleForSerialization
  Artifact getOutput() {
    return Iterables.getOnlyElement(outputs);
  }

  private static class ParamFileWriter implements DeterministicWriter {
    private final Iterable<String> arguments;
    private final ParameterFileType type;

    ParamFileWriter(Iterable<String> arguments, ParameterFileType type) {
      this.arguments = arguments;
      this.type = type;
    }

    @Override
    public void writeOutputFile(OutputStream out) throws IOException {
      ParameterFile.writeParameterFile(out, arguments, type, ISO_8859_1);
    }
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp)
      throws CommandLineExpansionException {
    fp.addString(GUID);
    fp.addString(String.valueOf(makeExecutable));
    fp.addString(type.toString());
    commandLine.addToFingerprint(actionKeyContext, fp);
  }
}
