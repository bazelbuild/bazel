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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.common.hash.HashCode;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
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
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collection;
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

    // Check that the primary output, :foo0.main, has its metadata retained.
    Artifact main =
        ((RuleConfiguredTarget) fooTarget)
            .findArtifactByOutputLabel(
                Label.parseCanonicalUnchecked("//validation_actions:foo0.main"));
    FileArtifactValue mainMetadata =
        targetCompleteEventRef.get().getCompletionContext().getFileArtifactValue(main);
    assertThat(mainMetadata).isNotNull();

    // Check that the validation output, :foo0.validation, does not have its metadata retained.
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
  public void outputFile() throws Exception {
    write("foo/BUILD", "genrule(name = 'foobin', outs = ['out.txt'], cmd = 'echo -n Hello > $@')");

    File bep = buildTargetAndCaptureBEP("//foo:foobin");

    BuildEventStreamProtos.File outFile = findOutputFileInBEPStream(bep, "out.txt");
    assertThat(outFile).isNotNull();
    assertThat(outFile.getUri()).startsWith("file://");
    assertThat(outFile.getUri()).endsWith("/bin/foo/out.txt");
    assertThat(outFile.getLength()).isEqualTo("Hello".length());
    assertDigest("Hello", BaseEncoding.base16().lowerCase().decode(outFile.getDigest()));
  }

  @Test
  public void outputDirectory() throws Exception {
    write(
        "foo/defs.bzl",
        "def _impl(ctx):",
        "  dir = ctx.actions.declare_directory(ctx.label.name)",
        "  ctx.actions.run_shell(",
        "    outputs = [dir],",
        "    command = 'echo -n Hello > %s/file.txt' % dir.path,",
        ")",
        "  return DefaultInfo(files = depset([dir]))",
        "",
        "directory = rule(implementation = _impl)");
    write("foo/BUILD", "load(':defs.bzl', 'directory')", "directory(name = 'dir')");

    File bep = buildTargetAndCaptureBEP("//foo:dir");

    BuildEventStreamProtos.TargetComplete targetComplete = findTargetCompleteEventInBEPStream(bep);
    assertThat(targetComplete.getDirectoryOutputList()).hasSize(1);
    BuildEventStreamProtos.File dir = targetComplete.getDirectoryOutputList().get(0);
    assertThat(dir.getName()).endsWith("/dir");
    assertThat(dir.getUri()).isEmpty();
    assertThat(dir.getContents()).isEmpty();
    assertThat(dir.getSymlinkTargetPath()).isEmpty();

    BuildEventStreamProtos.File outFile = findOutputFileInBEPStream(bep, "file.txt");
    assertThat(outFile).isNotNull();
    assertThat(outFile.getUri()).startsWith("file://");
    assertThat(outFile.getUri()).endsWith("/bin/foo/dir/file.txt");
    assertThat(outFile.getLength()).isEqualTo("Hello".length());
    assertDigest("Hello", BaseEncoding.base16().lowerCase().decode(outFile.getDigest()));
  }

  @Test
  public void outputSymlink() throws Exception {
    write(
        "foo/defs.bzl",
        "def _impl(ctx):",
        "  sym = ctx.actions.declare_symlink(ctx.label.name)",
        "  ctx.actions.symlink(output = sym, target_path = '/some/path')",
        "  return DefaultInfo(files = depset([sym]))",
        "",
        "symlink = rule(implementation = _impl)");
    write("foo/BUILD", "load(':defs.bzl', 'symlink')", "symlink(name = 'sym')");

    File bep = buildTargetAndCaptureBEP("//foo:sym");

    BuildEventStreamProtos.File outFile = findOutputFileInBEPStream(bep, "sym");
    assertThat(outFile).isNotNull();
    assertThat(outFile.getSymlinkTargetPath()).isEqualTo("/some/path");
    assertThat(outFile.getLength()).isEqualTo(0);
    assertThat(outFile.getDigest()).isEmpty();
  }

  private File buildTargetAndCaptureBEP(String target) throws Exception {
    File bep = tmpFolder.newFile();
    // We use WAIT_FOR_UPLOAD_COMPLETE because it's the easiest way to force the BES module to
    // wait until the BEP binary file has been written.
    addOptions(
        "--build_event_binary_file=" + bep.getAbsolutePath(),
        "--bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE");
    buildTarget(target);
    // We need to wait for all events to be written to the file, which is done in #afterCommand()
    // if --bes_upload_mode=WAIT_FOR_UPLOAD_COMPLETE.
    afterBuildCommand();
    return bep;
  }

  private static void assertDigest(String contents, byte[] bepDigest) {
    // Try all registered hash functions and verify that one of them was used to produce the digest.
    boolean foundHashFunction = false;
    for (DigestHashFunction hashFunction : DigestHashFunction.getPossibleHashFunctions()) {
      HashCode hashCode = hashFunction.getHashFunction().hashString(contents, UTF_8);
      if (Arrays.equals(bepDigest, hashCode.asBytes())) {
        foundHashFunction = true;
      }
    }
    assertThat(foundHashFunction).isTrue();
  }

  private static ImmutableList<BuildEvent> parseBuildEventsFromBEPStream(File bep)
      throws IOException {
    ImmutableList.Builder<BuildEvent> buildEvents = ImmutableList.builder();
    try (InputStream in = new FileInputStream(bep)) {
      while (in.available() > 0) {
        buildEvents.add(BuildEvent.parseDelimitedFrom(in));
      }
    }
    return buildEvents.build();
  }

  @Nullable
  private static BuildEventStreamProtos.TargetComplete findTargetCompleteEventInBEPStream(File bep)
      throws IOException {
    for (BuildEvent buildEvent : parseBuildEventsFromBEPStream(bep)) {
      if (buildEvent.getId().getIdCase() == IdCase.TARGET_COMPLETED) {
        return buildEvent.getCompleted();
      }
    }
    return null;
  }

  @Nullable
  private static BuildEventStreamProtos.File findOutputFileInBEPStream(File bep, String name)
      throws IOException {
    for (BuildEvent buildEvent : parseBuildEventsFromBEPStream(bep)) {
      if (buildEvent.getId().getIdCase() == IdCase.NAMED_SET) {
        NamedSetOfFiles namedSetOfFiles = buildEvent.getNamedSetOfFiles();
        for (BuildEventStreamProtos.File file : namedSetOfFiles.getFilesList()) {
          if (file.getName().contains(name)) {
            return file;
          }
        }
      }
    }
    return null;
  }
}
