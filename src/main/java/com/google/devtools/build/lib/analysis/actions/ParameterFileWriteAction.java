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
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ShellEscaper;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;

/**
 * Action to write a parameter file for a {@link CommandLine}.
 */
public final class ParameterFileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "45f678d8-e395-401e-8446-e795ccc6361f";

  private final CommandLine commandLine;
  private final ParameterFileType type;
  private final Charset charset;

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
    super(owner, ImmutableList.<Artifact>of(), output, false);
    this.commandLine = commandLine;
    this.type = type;
    this.charset = charset;
  }

  /**
   * Returns the list of options written to the parameter file. Don't use this
   * method outside tests - the list is often huge, resulting in significant
   * garbage collection overhead.
   */
  @VisibleForTesting
  public Iterable<String> getContents() {
    return commandLine.arguments();
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        switch (type) {
          case SHELL_QUOTED :
            writeContentQuoted(out);
            break;
          case UNQUOTED :
            writeContentUnquoted(out);
            break;
          default :
            throw new AssertionError();
        }
      }
    };
  }

  /**
   * Writes the arguments from the list into the parameter file.
   */
  private void writeContentUnquoted(OutputStream outputStream) throws IOException {
    OutputStreamWriter out = new OutputStreamWriter(outputStream, charset);
    for (String line : commandLine.arguments()) {
      out.write(line);
      out.write('\n');
    }
    out.flush();
  }

  /**
   * Writes the arguments from the list into the parameter file with shell
   * quoting (if required).
   */
  private void writeContentQuoted(OutputStream outputStream) throws IOException {
    OutputStreamWriter out = new OutputStreamWriter(outputStream, charset);
    for (String line : ShellEscaper.escapeAll(commandLine.arguments())) {
      out.write(line);
      out.write('\n');
    }
    out.flush();
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(String.valueOf(makeExecutable));
    f.addStrings(commandLine.arguments());
    return f.hexDigestAndReset();
  }
}
