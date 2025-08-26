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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ArgChunk;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.util.DeterministicWriter;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/** Action to write a parameter file for a {@link CommandLine}. */
@Immutable // if commandLine is immutable
public final class ParameterFileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "45f678d8-e395-401e-8446-e795ccc6361f";

  private final CommandLine commandLine;
  private final ParameterFileType type;
  private final boolean makeExecutable;
  private final String mnemonic;
  private final boolean usePathStripping;

  /**
   * Creates a new instance.
   *
   * @param owner the action owner
   * @param output the Artifact that will be created by executing this Action
   * @param commandLine the contents to be written to the file
   * @param type the type of the file
   * @param makeExecutable whether the output file should be made executable
   */
  public ParameterFileWriteAction(
      ActionOwner owner,
      Artifact output,
      CommandLine commandLine,
      ParameterFileType type,
      boolean makeExecutable) {
    this(
        owner,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        output,
        commandLine,
        type,
        makeExecutable,
        AbstractFileWriteAction.MNEMONIC,
        /* executionInfo= */ ImmutableMap.of(),
        CoreOptions.OutputPathsMode.OFF);
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
   * @param makeExecutable whether the output file should be made executable
   * @param mnemonic the mnemonic for this action, or null if the default should be used
   * @param executionInfo the execution info for this action (only supports-path-mapping is used)
   * @param outputPathsMode the output paths mode obtained via {@link
   *     PathMappers#getOutputPathsMode(BuildConfigurationValue)}
   */
  public ParameterFileWriteAction(
      ActionOwner owner,
      NestedSet<Artifact> inputs,
      Artifact output,
      CommandLine commandLine,
      ParameterFileType type,
      boolean makeExecutable,
      String mnemonic,
      ImmutableMap<String, String> executionInfo,
      CoreOptions.OutputPathsMode outputPathsMode) {
    super(owner, inputs, output);
    this.commandLine = commandLine;
    this.type = type;
    this.makeExecutable = makeExecutable;
    this.mnemonic = mnemonic;
    // Save memory by not storing the full execution info, but only what matters for this particular
    // action.
    this.usePathStripping =
        PathMappers.getEffectiveOutputPathsMode(outputPathsMode, getMnemonic(), executionInfo)
            == CoreOptions.OutputPathsMode.STRIP;
  }

  @Override
  public boolean makeExecutable() {
    return makeExecutable;
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  public ImmutableMap<String, String> getExecutionInfo() {
    return usePathStripping
        ? ImmutableMap.of(ExecutionRequirements.SUPPORTS_PATH_MAPPING, "")
        : ImmutableMap.of();
  }

  private CoreOptions.OutputPathsMode getOutputPathsMode() {
    return usePathStripping ? CoreOptions.OutputPathsMode.STRIP : CoreOptions.OutputPathsMode.OFF;
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
   *
   * <p>TODO(b/161359171): The list of arguments will be incorrect if the arguments contain tree
   * artifacts or path mapping is used.
   */
  public Iterable<String> getArguments()
      throws CommandLineExpansionException, InterruptedException {
    return commandLine.arguments();
  }

  @VisibleForTesting
  public String getStringContents()
      throws CommandLineExpansionException, InterruptedException, IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ParameterFile.writeParameterFile(out, getArguments(), type);
    return out.toString(ISO_8859_1);
  }

  @Nullable
  @Override
  public String getStarlarkContent() throws IOException, EvalException, InterruptedException {
    if (!getInputs().isEmpty()) {
      // Tree artifact information isn't available at analysis time.
      return null;
    }
    try {
      return getStringContents();
    } catch (CommandLineExpansionException e) {
      throw Starlark.errorf("Error expanding command line: %s", e.getMessage());
    }
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws ExecException, InterruptedException {
    final ArgChunk arguments;
    // Other actions consuming this parameter file may have path mapping disabled due to inputs
    // conflicting across configurations, in which case paths written to the file will not match.
    // Since this depends on the consumer but the decision is only made at execution time, it is not
    // clear how to improve that situation. Actions that are prone to such collisions should avoid
    // depending on parameter files.
    var pathMapper = PathMappers.create(this, getOutputPathsMode(), /* isStarlarkAction= */ false);
    try {
      InputMetadataProvider inputMetadataProvider =
          Preconditions.checkNotNull(ctx.getInputMetadataProvider());
      arguments = commandLine.expand(inputMetadataProvider, pathMapper);
    } catch (CommandLineExpansionException e) {
      throw new UserExecException(
          e,
          FailureDetail.newBuilder()
              .setMessage(Strings.nullToEmpty(e.getMessage()))
              .setSpawn(Spawn.newBuilder().setCode(Code.COMMAND_LINE_EXPANSION_FAILURE))
              .build());
    }
    return new ParamFileWriter(arguments, pathMapper, type);
  }

  private static class ParamFileWriter implements DeterministicWriter {
    private final ArgChunk arguments;
    private final PathMapper pathMapper;
    private final ParameterFileType type;

    ParamFileWriter(ArgChunk arguments, PathMapper pathMapper, ParameterFileType type) {
      this.arguments = arguments;
      this.pathMapper = pathMapper;
      this.type = type;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      ParameterFile.writeParameterFile(out, arguments.arguments(pathMapper), type);
    }
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    fp.addString(GUID);
    fp.addString(type.toString());
    commandLine.addToFingerprint(
        actionKeyContext,
        inputMetadataProvider,
        PathMappers.getEffectiveOutputPathsMode(
            getOutputPathsMode(), getMnemonic(), getExecutionInfo()),
        fp);
  }

  @Override
  public String describeKey() {
    StringBuilder message = new StringBuilder();
    message.append("GUID: ");
    message.append(GUID);
    message.append("\nParam File Type: ");
    message.append(type);
    message.append("\nContent digest (approximate): ");
    try {
      // The full contents can be huge, which makes the final error message
      // incomprehensible. Instead, just give a digest, which makes it easy to
      // tell if two contents are equal or not.
      var fp = new Fingerprint();
      commandLine.addToFingerprint(
          new ActionKeyContext(),
          null,
          PathMappers.getEffectiveOutputPathsMode(
              getOutputPathsMode(), getMnemonic(), getExecutionInfo()),
          fp);
      message.append(BaseEncoding.base16().lowerCase().encode(fp.digestAndReset()));
      message.append(
          "\n"
              + "NOTE: Content digest reflects approximate, analysis-time data; it does not account"
              + " for data available during execution (e.g. tree artifact expansions)");
    } catch (InterruptedException ex) {
      Thread.currentThread().interrupt();
      message.append("Interrupted while expanding command line");
    } catch (CommandLineExpansionException e) {
      message.append("Could not expand contents: ");
      message.append(e);
    }
    return message.toString();
  }
}
