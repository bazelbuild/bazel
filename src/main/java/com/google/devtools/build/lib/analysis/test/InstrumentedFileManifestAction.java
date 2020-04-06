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

package com.google.devtools.build.lib.analysis.test;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifacts;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Arrays;

/**
 * Writes a manifest of instrumented source and metadata files.
 */
@Immutable
final class InstrumentedFileManifestAction extends AbstractFileWriteAction {
  private static final String GUID = "3833f0a3-7ea1-4d9f-b96f-66eff4c922b0";

  private final NestedSet<Artifact> files;

  @VisibleForTesting
  InstrumentedFileManifestAction(ActionOwner owner, NestedSet<Artifact> files, Artifact output) {
    super(
        owner,
        /*inputs=*/ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        output,
        /*makeExecutable=*/ false);
    this.files = files;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        // Sort the exec paths before writing them out.
        String[] fileNames =
            files.toList().stream().map(Artifact::getExecPathString).toArray(String[]::new);
        Arrays.sort(fileNames);
        try (Writer writer = new OutputStreamWriter(out, ISO_8859_1)) {
          for (String name : fileNames) {
            writer.write(name);
            writer.write('\n');
          }
        }
      }
    };
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    // TODO(b/150305897): use addUUID?
    fp.addString(GUID);
    // TODO(b/150308417): Not sorting is probably cheaper, might lead to unnecessary re-execution.
    Artifacts.addToFingerprint(fp, files.toList());
  }

  /**
   * Instantiates instrumented file manifest for the given target.
   *
   * @param ruleContext context of the executable configured target
   * @param additionalSourceFiles additional instrumented source files, as
   *                              collected by the {@link InstrumentedFilesCollector}
   * @param metadataFiles *.gcno/*.em files collected by the {@link InstrumentedFilesCollector}
   * @return instrumented file manifest artifact
   */
  public static Artifact getInstrumentedFileManifest(RuleContext ruleContext,
      NestedSet<Artifact> additionalSourceFiles, NestedSet<Artifact> metadataFiles) {
    // Instrumented manifest makes sense only for rules with binary output.
    Preconditions.checkState(ruleContext.getRule().hasBinaryOutput());
    Artifact instrumentedFileManifest = ruleContext.getBinArtifact(
        ruleContext.getTarget().getName()  + ".instrumented_files");

    NestedSet<Artifact> inputs = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(additionalSourceFiles)
        .addTransitive(metadataFiles)
        .build();
    ruleContext.registerAction(new InstrumentedFileManifestAction(
        ruleContext.getActionOwner(), inputs, instrumentedFileManifest));

    return instrumentedFileManifest;
  }
}
