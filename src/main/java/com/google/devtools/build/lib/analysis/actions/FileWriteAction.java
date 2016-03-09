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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.util.Fingerprint;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

/**
 * Action to write to a file.
 * <p>TODO(bazel-team): Choose a better name to distinguish this class from
 * {@link BinaryFileWriteAction}.
 */
public class FileWriteAction extends AbstractFileWriteAction {

  private static final String GUID = "332877c7-ca9f-4731-b387-54f620408522";

  /**
   * We keep it as a CharSequence for memory-efficiency reasons. The toString()
   * method of the object represents the content of the file.
   *
   * <p>For example, this allows us to keep a {@code List<Artifact>} wrapped
   * in a {@code LazyString} instead of the string representation of the concatenation.
   * This saves memory because the Artifacts are shared objects but the
   * resulting String is not.
   */
  private final CharSequence fileContents;

  /**
   * Creates a new FileWriteAction instance without inputs using UTF8 encoding.
   *
   * @param owner the action owner.
   * @param output the Artifact that will be created by executing this Action.
   * @param fileContents the contents to be written to the file.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public FileWriteAction(ActionOwner owner, Artifact output, CharSequence fileContents,
      boolean makeExecutable) {
    this(owner, Artifact.NO_ARTIFACTS, output, fileContents, makeExecutable);
  }

  /**
   * Creates a new FileWriteAction instance using UTF8 encoding.
   *
   * @param owner the action owner.
   * @param inputs the Artifacts that this Action depends on
   * @param output the Artifact that will be created by executing this Action.
   * @param fileContents the contents to be written to the file.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public FileWriteAction(ActionOwner owner, Collection<Artifact> inputs,
      Artifact output, CharSequence fileContents, boolean makeExecutable) {
    super(owner, inputs, output, makeExecutable);
    this.fileContents = fileContents;
  }

  /**
   * Creates a new FileWriteAction instance using UTF8 encoding.
   *
   * @param owner the action owner.
   * @param inputs the Artifacts that this Action depends on
   * @param output the Artifact that will be created by executing this Action.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  protected FileWriteAction(ActionOwner owner, Collection<Artifact> inputs,
      Artifact output, boolean makeExecutable) {
    this(owner, inputs, output, "", makeExecutable);
  }

  public String getFileContents() {
    return fileContents.toString();
  }

  /**
   * Create a DeterministicWriter for the content of the output file as provided by
   * {@link #getFileContents()}.
   */
  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        out.write(getFileContents().getBytes(UTF_8));
      }
    };
  }

  /**
   * Computes the Action key for this action by computing the fingerprint for
   * the file contents.
   */
  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(String.valueOf(makeExecutable));
    f.addString(getFileContents());
    return f.hexDigestAndReset();
  }

  /**
   * Creates a FileWriteAction to write contents to the resulting artifact
   * fileName in the genfiles root underneath the package path.
   *
   * @param ruleContext the ruleContext that will own the action of creating this file.
   * @param fileName name of the file to create.
   * @param contents data to write to file.
   * @param executable flags that file should be marked executable.
   * @return Artifact describing the file to create.
   */
  public static Artifact createFile(RuleContext ruleContext,
      String fileName, CharSequence contents, boolean executable) {
    Artifact scriptFileArtifact = ruleContext.getPackageRelativeArtifact(
        fileName, ruleContext.getConfiguration().getGenfilesDirectory());
    ruleContext.registerAction(new FileWriteAction(
        ruleContext.getActionOwner(), scriptFileArtifact, contents, executable));
    return scriptFileArtifact;
  }
}
