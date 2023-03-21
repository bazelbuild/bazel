// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.common.hash.HashCode;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildeventservice.BazelBuildEventServiceModule;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.IdCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.NamedSetOfFiles;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.NoSpawnCacheModule;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Verifies TargetCompleteEvent behavior during a complete build. */
@RunWith(JUnit4.class)
public final class TargetCompleteEventTest extends BuildIntegrationTestCase {

  @Rule public final TemporaryFolder tmpFolder = new TemporaryFolder();

  @Before
  public void stageEmbeddedTools() throws Exception {
    AnalysisMock.get().setupMockToolsRepository(mockToolsConfig);
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new NoSpawnCacheModule())
        .addBlazeModule(new CredentialModule())
        .addBlazeModule(new BazelBuildEventServiceModule());
  }

  private void afterBuildCommand() throws Exception {
    runtimeWrapper.newCommand();
  }

  /**
   * Validates that TargetCompleteEvents do not keep a map of action output metadata for the
   * _validation output group, which can be quite large.
   */
  @Test
  public void artifactsNotRetained() throws Exception {
    write(
        "validation_actions/defs.bzl",
        "def _rule_with_implicit_outs_and_validation_impl(ctx):",
        "",
        "  ctx.actions.write(ctx.outputs.main, \"main output\\n\")",
        "",
        "  ctx.actions.write(ctx.outputs.implicit, \"implicit output\\n\")",
        "",
        "  validation_output = ctx.actions.declare_file(ctx.attr.name + \".validation\")",
        "  # The actual tool will be created in individual tests, depending on whether",
        "  # validation should pass or fail.",
        "  ctx.actions.run(",
        "      outputs = [validation_output],",
        "      executable = ctx.executable._validation_tool,",
        "      arguments = [validation_output.path])",
        "",
        "  return [",
        "    DefaultInfo(files = depset([ctx.outputs.main])),",
        "    OutputGroupInfo(_validation = depset([validation_output])),",
        "  ]",
        "",
        "",
        "rule_with_implicit_outs_and_validation = rule(",
        "  implementation = _rule_with_implicit_outs_and_validation_impl,",
        "  outputs = {",
        "    \"main\": \"%{name}.main\",",
        "    \"implicit\": \"%{name}.implicit\",",
        "  },",
        "  attrs = {",
        "    \"_validation_tool\": attr.label(",
        "        allow_single_file = True,",
        "        default = Label(\"//validation_actions:validation_tool\"),",
        "        executable = True,",
        "        cfg = \"exec\"),",
        "  }",
        ")");
    write("validation_actions/validation_tool", "#!/bin/bash", "echo \"validation output\" > $1")
        .setExecutable(true);
    write(
        "validation_actions/BUILD",
        "load(",
        "    \":defs.bzl\",",
        "    \"rule_with_implicit_outs_and_validation\")",
        "",
        "rule_with_implicit_outs_and_validation(name = \"foo0\")");

    AtomicReference<TargetCompleteEvent> targetCompleteEventRef = new AtomicReference<>();
    runtimeWrapper.registerSubscriber(
        new Object() {
          @SuppressWarnings("unused")
          @Subscribe
          public void accept(TargetCompleteEvent event) {
            targetCompleteEventRef.set(event);
          }
        });

    addOptions("--experimental_run_validations");
    BuildResult buildResult = buildTarget("//validation_actions:foo0");

    Collection<ConfiguredTarget> successfulTargets = buildResult.getSuccessfulTargets();
    ConfiguredTarget fooTarget = Iterables.getOnlyElement(successfulTargets);

    // Check that the primary output, :foo0.main, has its metadata retained and the
    // CompletionContext can confirm it is an output file.
    Artifact main =
        ((RuleConfiguredTarget) fooTarget)
            .getArtifactByOutputLabel(
                Label.parseCanonicalUnchecked("//validation_actions:foo0.main"));
    FileStateType mainType =
        targetCompleteEventRef.get().getCompletionContext().getFileArtifactValue(main).getType();
    assertThat(CompletionContext.isGuaranteedToBeOutputFile(mainType)).isTrue();

    // Check that the validation output, :foo0.validation, does not have its metadata retained and
    // the CompletionContext cannot confirm it is an output file (even though it is).
    OutputGroupInfo outputGroups = fooTarget.get(OutputGroupInfo.STARLARK_CONSTRUCTOR);
    NestedSet<Artifact> validationArtifacts =
        outputGroups.getOutputGroup(OutputGroupInfo.VALIDATION);
    assertThat(validationArtifacts.isEmpty()).isFalse();

    Artifact validationArtifact = Iterables.getOnlyElement(validationArtifacts.toList());

    FileArtifactValue validationArtifactMetadata =
        targetCompleteEventRef
            .get()
            .getCompletionContext()
            .getFileArtifactValue(validationArtifact);
    assertThat(validationArtifactMetadata).isNull();
  }

  @Test
  public void digestAndLengthInBuildEventProtocol() throws Exception {
    // Produces a TargetCompleteEvent in BEP and verifies that we include the output file's
    // length and digest.
    write(
        "foo/BUILD",
        "genrule(name = 'foobin', outs = ['out.txt'], cmd = 'echo -n \"Hello\" > $@')");
    File buildEventBinaryFile = tmpFolder.newFile();
    // We use WAIT_FOR_UPLOAD_COMPLETE because it's the easiest way to force the BES module to
    // wait until the BEP binary file has been written.
    addOptions(
        "--build_event_binary_file=" + buildEventBinaryFile.getAbsolutePath(),
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");
    buildTarget("//foo:foobin");
    // We need to wait for all events to be written to the file, which is done in #afterCommand()
    // if --bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE.
    afterBuildCommand();

    List<BuildEvent> buildEvents = new ArrayList<>();
    try (InputStream in = new FileInputStream(buildEventBinaryFile)) {
      while (in.available() > 0) {
        buildEvents.add(BuildEvent.parseDelimitedFrom(in));
      }
    }
    BuildEventStreamProtos.File outFile = findOutputFileInBEPStream(buildEvents);
    assertThat(outFile).isNotNull();
    assertThat(outFile.getLength()).isEqualTo("Hello".length());
    byte[] bepDigest = BaseEncoding.base16().lowerCase().decode(outFile.getDigest());
    // Try all registered hash functions and verify that one of them was used to produce the digest.
    boolean foundHashFunction = false;
    for (DigestHashFunction hashFunction : DigestHashFunction.getPossibleHashFunctions()) {
      HashCode hashCode =
          hashFunction.getHashFunction().hashString("Hello", StandardCharsets.UTF_8);
      if (Arrays.equals(bepDigest, hashCode.asBytes())) {
        foundHashFunction = true;
      }
    }
    assertThat(foundHashFunction).isTrue();
  }

  @Nullable
  private static BuildEventStreamProtos.File findOutputFileInBEPStream(
      List<BuildEvent> buildEvents) {
    for (BuildEvent buildEvent : buildEvents) {
      if (buildEvent.getId().getIdCase() == IdCase.NAMED_SET) {
        NamedSetOfFiles namedSetOfFiles = buildEvent.getNamedSetOfFiles();
        for (BuildEventStreamProtos.File file : namedSetOfFiles.getFilesList()) {
          if (file.getName().contains("out.txt")) {
            return file;
          }
        }
      }
    }
    return null;
  }
}
