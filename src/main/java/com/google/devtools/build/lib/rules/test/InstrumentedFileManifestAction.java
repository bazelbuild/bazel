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

package com.google.devtools.build.lib.rules.test;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.RegexFilter;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.Collection;

/**
 * Creates instrumented file manifest to list instrumented source files.
 */
class InstrumentedFileManifestAction extends AbstractFileWriteAction {

  private static final String GUID = "d9ddb800-f9a1-01Da-238d-988311a8475b";

  private final Collection<Artifact> collectedSourceFiles;
  private final Collection<Artifact> metadataFiles;
  private final RegexFilter instrumentationFilter;

  private InstrumentedFileManifestAction(ActionOwner owner, Collection<Artifact> inputs,
      Collection<Artifact> additionalSourceFiles, Collection<Artifact> gcnoFiles,
      Artifact output, RegexFilter instrumentationFilter) {
    super(owner, inputs, output, false);
    this.collectedSourceFiles = additionalSourceFiles;
    this.metadataFiles = gcnoFiles;
    this.instrumentationFilter = instrumentationFilter;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        Writer writer = null;
        try {
          // Save exec paths for both instrumented source files and gcno files in the manifest
          // in the naturally sorted order.
          String[] fileNames = Iterables.toArray(Iterables.transform(
              Iterables.concat(collectedSourceFiles, metadataFiles),
              new Function<Artifact, String> () {
                @Override
                public String apply(Artifact artifact) { return artifact.getExecPathString(); }
              }), String.class);
          Arrays.sort(fileNames);
          writer = new OutputStreamWriter(out, ISO_8859_1);
          for (String name : fileNames) {
            writer.write(name);
            writer.write('\n');
          }
        } finally {
          if (writer != null) {
            writer.close();
          }
        }
      }
    };
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(instrumentationFilter.toString());
    return f.hexDigestAndReset();
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
  public static Artifact getInstrumentedFileManifest(final RuleContext ruleContext,
      final Collection<Artifact> additionalSourceFiles, final Collection<Artifact> metadataFiles) {
    // Instrumented manifest makes sense only for rules with binary output.
    Preconditions.checkState(ruleContext.getRule().hasBinaryOutput());
    Artifact instrumentedFileManifest = ruleContext.getPackageRelativeArtifact(
        ruleContext.getTarget().getName()  + ".instrumented_files",
        ruleContext.getConfiguration().getBinDirectory());

    // Instrumented manifest artifact might already exist in case when multiple test
    // actions that use slightly different subsets of runfiles set are generated for the same rule.
    // So check whether we need to create a new action instance.
    ImmutableList<Artifact> inputs = ImmutableList.<Artifact>builder()
        .addAll(additionalSourceFiles)
        .addAll(metadataFiles)
        .build();
    ruleContext.registerAction(new InstrumentedFileManifestAction(
        ruleContext.getActionOwner(), inputs, additionalSourceFiles, metadataFiles,
        instrumentedFileManifest, ruleContext.getConfiguration().getInstrumentationFilter()));

    return instrumentedFileManifest;
  }
}
