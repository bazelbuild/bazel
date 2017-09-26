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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ShellEscaper;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;

/**
 * Action to write a parameter file for a {@link CommandLine}.
 */
@Immutable // if commandLine and charset are immutable
public final class ParameterFileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "45f678d8-e395-401e-8446-e795ccc6361f";

  private final CommandLine commandLine;
  private final ParameterFileType type;
  private final Charset charset;
  private final boolean hasInputArtifactToExpand;

  /**
   * Creates a new instance.
   *
   * @param owner the action owner
   * @param output the Artifact that will be created by executing this Action
   * @param commandLine the contents to be written to the file
   * @param type the type of the file
   * @param charset the charset of the file
   */
  public ParameterFileWriteAction(ActionOwner owner, Artifact output, CommandLine commandLine,
      ParameterFileType type, Charset charset) {
    this(owner, ImmutableList.<Artifact>of(), output, commandLine, type, charset);
  }

  /**
   * Creates a new instance.
   *
   * @param owner the action owner
   * @param inputs the list of TreeArtifacts that must be resolved and expanded before evaluating
   *     the contents of {@link commandLine}.
   * @param output the Artifact that will be created by executing this Action
   * @param commandLine the contents to be written to the file
   * @param type the type of the file
   * @param charset the charset of the file
   */
  public ParameterFileWriteAction(ActionOwner owner, Iterable<Artifact> inputs, Artifact output,
      CommandLine commandLine, ParameterFileType type, Charset charset) {
    super(owner, ImmutableList.copyOf(inputs), output, false);
    this.commandLine = commandLine;
    this.type = type;
    this.charset = charset;
    this.hasInputArtifactToExpand = !Iterables.isEmpty(inputs);
  }

  /**
   * Returns the list of options written to the parameter file. Don't use this method outside tests
   * - the list is often huge, resulting in significant garbage collection overhead.
   */
  @VisibleForTesting
  public Iterable<String> getContents() throws CommandLineExpansionException {
    Preconditions.checkState(
        !hasInputArtifactToExpand,
        "This action contains a CommandLine with TreeArtifacts: %s, which must be expanded using "
        + "ArtifactExpander first before we can evaluate the CommandLine.",
        getInputs());
    return commandLine.arguments();
  }

  @VisibleForTesting
  public Iterable<String> getContents(ArtifactExpander artifactExpander)
      throws CommandLineExpansionException {
    return commandLine.arguments(artifactExpander);
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
    return new ParamFileWriter(arguments);
  }

  private class ParamFileWriter implements DeterministicWriter {
    private final Iterable<String> arguments;

    ParamFileWriter(Iterable<String> arguments) {
      this.arguments = arguments;
    }

    @Override
    public void writeOutputFile(OutputStream out) throws IOException {
      switch (type) {
        case SHELL_QUOTED:
          writeContentQuoted(out, arguments);
          break;
        case UNQUOTED:
          writeContentUnquoted(out, arguments);
          break;
        default:
          throw new AssertionError();
      }
    }

    /**
     * Writes the arguments from the list into the parameter file.
     */
    private void writeContentUnquoted(OutputStream outputStream, Iterable<String> arguments)
        throws IOException {
      OutputStreamWriter out = new OutputStreamWriter(outputStream, charset);
      for (String line : arguments) {
        out.write(line);
        out.write('\n');
      }
      out.flush();
    }

    /**
     * Writes the arguments from the list into the parameter file with shell
     * quoting (if required).
     */
    private void writeContentQuoted(OutputStream outputStream, Iterable<String> arguments)
        throws IOException {
      OutputStreamWriter out = new OutputStreamWriter(outputStream, charset);
      for (String line : ShellEscaper.escapeAll(arguments)) {
        out.write(line);
        out.write('\n');
      }
      out.flush();
    }
  }

  @Override
  protected String computeKey() throws CommandLineExpansionException {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(String.valueOf(makeExecutable));
    f.addString(type.toString());
    f.addString(charset.toString());
    f.addStrings(commandLine.arguments());
    return f.hexDigestAndReset();
  }
}
