// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CompletionContext;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsInOutputGroup;
import com.google.devtools.build.lib.analysis.TopLevelArtifactHelper.ArtifactsToBuild;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext.OutputGroupFileMode;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions.OutputGroupFileModes;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.File;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.OutputGroup;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TargetComplete;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.common.options.OptionsParser;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TargetCompleteEvent}. */
@RunWith(JUnit4.class)
public class TargetCompleteEventTest extends AnalysisTestCase {

  @Test
  public void testReferencedSourceFile() throws Exception {
    scratch.file("BUILD", "filegroup(name = 'files', srcs = ['file'])");
    scratch.file("file", "content does not matter");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:files");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    Artifact artifact = Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);
    CompletionContext completionContext =
        getCompletionContext(ImmutableMap.of(artifact, metadata), ImmutableMap.of());

    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            completionContext,
            artifactsToBuild.getAllArtifactsByOutputGroup(),
            /* announceTargetSummary= */ false);

    assertThat(event.referencedLocalFiles())
        .containsExactly(new LocalFile(artifact.getPath(), LocalFileType.OUTPUT_FILE, metadata));
  }

  @Test
  public void testReferencedLocalFiles_noneWhenTheProtoNamesFilesThroughNamedSets()
      throws Exception {
    TargetCompleteEvent event = eventForOneSourceFile();

    assertThat(event.referencedLocalFiles(bepOptions())).isEmpty();
  }

  @Test
  public void testReferencedLocalFiles_allWithLegacyImportantOutputs() throws Exception {
    TargetCompleteEvent event = eventForOneSourceFile();

    assertThat(event.referencedLocalFiles(bepOptions("--legacy_important_outputs")))
        .containsExactlyElementsIn(event.referencedLocalFiles());
    assertThat(event.referencedLocalFiles()).hasSize(1);
  }

  @Test
  public void testReferencedLocalFiles_thoseOfOutputGroupsReportedInline() throws Exception {
    TargetCompleteEvent event = eventForOneSourceFile();

    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=default=inline_only")))
        .containsExactlyElementsIn(event.referencedLocalFiles());
    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=default=both")))
        .containsExactlyElementsIn(event.referencedLocalFiles());
    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=other=inline_only")))
        .isEmpty();
  }

  @Test
  public void testReferencedLocalFiles_onlyThoseOfTheOutputGroupsReportedInline()
      throws Exception {
    TwoOutputGroups target = targetWithTwoOutputGroups();
    TargetCompleteEvent event = target.targetCompleteEvent();

    assertThat(event.referencedLocalFiles(bepOptions())).isEmpty();
    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=extra=inline_only")))
        .containsExactly(target.extraFile);
    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=default=both")))
        .containsExactly(target.defaultFile);
    assertThat(
            event.referencedLocalFiles(
                bepOptions(
                    "--experimental_build_event_output_group_mode=default=inline_only",
                    "--experimental_build_event_output_group_mode=extra=both")))
        .containsExactly(target.defaultFile, target.extraFile);
    assertThat(
            event.referencedLocalFiles(
                bepOptions(
                    "--legacy_important_outputs",
                    "--experimental_build_event_output_group_mode=extra=inline_only")))
        .containsExactly(target.defaultFile, target.extraFile);
  }

  @Test
  public void testReferencedLocalFiles_ofATreeArtifactReportedInline() throws Exception {
    scratch.file(
        "defs.bzl",
        """
        def _impl(ctx):
            d = ctx.actions.declare_directory(ctx.label.name)
            ctx.actions.run_shell(outputs = [d], command = "does not matter")
            return DefaultInfo(files = depset([d]))

        dir = rule(_impl)
        """);
    scratch.file("BUILD", "load(':defs.bzl', 'dir')", "dir(name = 'dir')");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:dir");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    TreeArtifact tree =
        (TreeArtifact) Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    TreeFileArtifact child =
        TreeFileArtifact.createTreeOutput(tree, PathFragment.create("file.txt"));
    FileArtifactValue childMetadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);
    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            getCompletionContext(
                ImmutableMap.of(),
                ImmutableMap.of(
                    tree, TreeArtifactValue.newBuilder(tree).putChild(child, childMetadata).build())),
            artifactsToBuild.getAllArtifactsByOutputGroup(),
            /* announceTargetSummary= */ false);

    assertThat(event.referencedLocalFiles(bepOptions())).isEmpty();
    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=default=inline_only")))
        .containsExactly(new LocalFile(child.getPath(), LocalFileType.OUTPUT_FILE, childMetadata));
  }

  /**
   * The files that an event names are decided where its representation is written, and the files
   * it declares where a transport asks for them. A transport cannot give the location of a file
   * that was not declared, so the two have to agree under every combination of the options.
   */
  @Test
  public void testProtoNamesOnlyTheReferencedLocalFiles() throws Exception {
    TwoOutputGroups target = targetWithTwoOutputGroups();
    ImmutableList<ImmutableList<String>> optionsToTry =
        ImmutableList.of(
            ImmutableList.of(),
            ImmutableList.of("--legacy_important_outputs"),
            ImmutableList.of("--experimental_build_event_output_group_mode=default=inline_only"),
            ImmutableList.of("--experimental_build_event_output_group_mode=extra=both"),
            ImmutableList.of(
                "--experimental_build_event_output_group_mode=default=both",
                "--experimental_build_event_output_group_mode=extra=inline_only"),
            ImmutableList.of(
                "--legacy_important_outputs",
                "--experimental_build_event_output_group_mode=extra=inline_only"));

    for (ImmutableList<String> args : optionsToTry) {
      BuildEventProtocolOptions options = bepOptions(args.toArray(new String[0]));
      for (BuildEvent event :
          ImmutableList.of(target.targetCompleteEvent(), target.aspectCompleteEvent())) {
        ImmutableSet<String> declared =
            event.referencedLocalFiles(options).stream()
                .map(file -> uri(file.path))
                .collect(toImmutableSet());

        // Throws if the event names a file that it did not declare.
        TargetComplete proto =
            event.asStreamProto(contextKnowingOnly(declared, options)).getCompleted();

        ImmutableSet.Builder<String> named = ImmutableSet.builder();
        proto.getImportantOutputList().forEach(file -> named.add(file.getUri()));
        for (OutputGroup group : proto.getOutputGroupList()) {
          group.getInlineFilesList().forEach(file -> named.add(file.getUri()));
        }
        assertWithMessage("%s with %s", event.getClass().getSimpleName(), args)
            .that(named.build())
            .containsExactlyElementsIn(declared);
      }
    }
  }

  @Test
  public void testAspectCompleteEvent_referencesTheFilesOfOutputGroupsReportedInline()
      throws Exception {
    TwoOutputGroups target = targetWithTwoOutputGroups();
    AspectCompleteEvent event = target.aspectCompleteEvent();

    assertThat(event.referencedLocalFiles(bepOptions())).isEmpty();
    // An aspect has no legacy field for important outputs.
    assertThat(event.referencedLocalFiles(bepOptions("--legacy_important_outputs"))).isEmpty();
    assertThat(
            event.referencedLocalFiles(
                bepOptions("--experimental_build_event_output_group_mode=extra=inline_only")))
        .containsExactly(target.extraFile);
  }

  /** A target with one generated file in each of the output groups "default" and "extra". */
  private record TwoOutputGroups(
      ConfiguredTargetAndData ctAndData,
      CompletionContext completionContext,
      ImmutableMap<String, ArtifactsInOutputGroup> outputs,
      LocalFile defaultFile,
      LocalFile extraFile) {
    TargetCompleteEvent targetCompleteEvent() {
      return TargetCompleteEvent.successfulBuild(
          ctAndData, completionContext, outputs, /* announceTargetSummary= */ false);
    }

    AspectCompleteEvent aspectCompleteEvent() {
      return AspectCompleteEvent.createSuccessful(
          AspectKeyCreator.createAspectKey(
              AspectDescriptor.of(() -> "some_aspect", AspectParameters.EMPTY),
              ConfiguredTargetKey.fromConfiguredTarget(ctAndData.getConfiguredTarget())),
          completionContext,
          outputs,
          /* printToMasterLog= */ false);
    }
  }

  private TwoOutputGroups targetWithTwoOutputGroups() throws Exception {
    scratch.file(
        "defs.bzl",
        """
        def _impl(ctx):
            default = ctx.actions.declare_file(ctx.label.name + ".default")
            extra = ctx.actions.declare_file(ctx.label.name + ".extra")
            ctx.actions.write(default, "does not matter")
            ctx.actions.write(extra, "does not matter")
            return [
                DefaultInfo(files = depset([default])),
                OutputGroupInfo(extra = depset([extra])),
            ]

        two_groups = rule(_impl)
        """);
    scratch.file("BUILD", "load(':defs.bzl', 'two_groups')", "two_groups(name = 'two')");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:two");
    ImmutableMap<String, ArtifactsInOutputGroup> outputs =
        getArtifactsToBuild(ctAndData, ImmutableSortedSet.of("default", "extra"))
            .getAllArtifactsByOutputGroup();
    Artifact defaultArtifact = outputs.get("default").getArtifacts().getSingleton();
    Artifact extraArtifact = outputs.get("extra").getArtifacts().getSingleton();
    FileArtifactValue defaultMetadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);
    FileArtifactValue extraMetadata =
        FileArtifactValue.createForNormalFile(new byte[] {4, 5, 6}, null, 20);
    return new TwoOutputGroups(
        ctAndData,
        getCompletionContext(
            ImmutableMap.of(defaultArtifact, defaultMetadata, extraArtifact, extraMetadata),
            ImmutableMap.of()),
        outputs,
        new LocalFile(defaultArtifact.getPath(), LocalFileType.OUTPUT_FILE, defaultMetadata),
        new LocalFile(extraArtifact.getPath(), LocalFileType.OUTPUT_FILE, extraMetadata));
  }

  private static String uri(Path path) {
    return "file://" + path.getPathString();
  }

  /**
   * Returns a context that gives the location of the given files, as a transport does of those
   * that an event declared, and throws when asked for that of another.
   */
  private static BuildEventContext contextKnowingOnly(
      ImmutableSet<String> uris, BuildEventProtocolOptions options) {
    OutputGroupFileModes modes = options.getOutputGroupFileModesMapping();
    return new BuildEventContext() {
      @Override
      public PathConverter pathConverter() {
        return path -> {
          String uri = uri(path);
          if (!uris.contains(uri)) {
            throw new IllegalStateException("Illegal file reference: " + path);
          }
          return uri;
        };
      }

      @Override
      public ArtifactGroupNamer artifactGroupNamer() {
        return node -> NamedSetOfFilesId.newBuilder().setId("some set").build();
      }

      @Override
      public BuildEventProtocolOptions getOptions() {
        return options;
      }

      @Override
      public OutputGroupFileMode getFileModeForOutputGroup(String outputGroup) {
        return modes.getMode(outputGroup);
      }
    };
  }

  private TargetCompleteEvent eventForOneSourceFile() throws Exception {
    scratch.file("BUILD", "filegroup(name = 'files', srcs = ['file'])");
    scratch.file("file", "content does not matter");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:files");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    Artifact artifact = Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);
    return TargetCompleteEvent.successfulBuild(
        ctAndData,
        getCompletionContext(ImmutableMap.of(artifact, metadata), ImmutableMap.of()),
        artifactsToBuild.getAllArtifactsByOutputGroup(),
        /* announceTargetSummary= */ false);
  }

  private static BuildEventProtocolOptions bepOptions(String... args) throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildEventProtocolOptions.class).build();
    parser.parse(args);
    return parser.getOptions(BuildEventProtocolOptions.class);
  }

  @Test
  public void testReferencedSourceDirectory() throws Exception {
    scratch.file("BUILD", "filegroup(name = 'files', srcs = ['dir'])");
    scratch.file("dir/file", "content does not matter");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:files");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    Artifact artifact = Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    FileArtifactValue metadata = FileArtifactValue.createForDirectoryWithMtime(0);
    CompletionContext completionContext =
        getCompletionContext(ImmutableMap.of(artifact, metadata), ImmutableMap.of());

    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            completionContext,
            artifactsToBuild.getAllArtifactsByOutputGroup(),
            /* announceTargetSummary= */ false);

    assertThat(event.referencedLocalFiles())
        .containsExactly(
            new LocalFile(artifact.getPath(), LocalFileType.OUTPUT_DIRECTORY, metadata));
  }

  @Test
  public void testReferencedTreeArtifact() throws Exception {
    scratch.file(
        "defs.bzl",
        """
        def _impl(ctx):
            d = ctx.actions.declare_directory(ctx.label.name)
            ctx.actions.run_shell(outputs = [d], command = "does not matter")
            return DefaultInfo(files = depset([d]))

        dir = rule(_impl)
        """);
    scratch.file(
        "BUILD",
        "load(':defs.bzl', 'dir')",
        "dir(name = 'dir')",
        "filegroup(name = 'files', srcs = ['dir'])");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:files");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    TreeArtifact tree =
        (TreeArtifact) Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    TreeFileArtifact fileChild =
        TreeFileArtifact.createTreeOutput(tree, PathFragment.create("dir/file.txt"));
    FileArtifactValue fileMetadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);
    // A TreeFileArtifact can be a directory, when materialized by a symlink.
    // See https://github.com/bazelbuild/bazel/issues/20418.
    TreeFileArtifact dirChild = TreeFileArtifact.createTreeOutput(tree, PathFragment.create("sym"));
    FileArtifactValue dirMetadata = FileArtifactValue.createForDirectoryWithMtime(123456789);
    TreeArtifactValue metadata =
        TreeArtifactValue.newBuilder(tree)
            .putChild(fileChild, fileMetadata)
            .putChild(dirChild, dirMetadata)
            .build();
    CompletionContext completionContext =
        getCompletionContext(ImmutableMap.of(), ImmutableMap.of(tree, metadata));

    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            completionContext,
            artifactsToBuild.getAllArtifactsByOutputGroup(),
            /* announceTargetSummary= */ false);

    assertThat(event.referencedLocalFiles())
        .containsExactly(
            new LocalFile(fileChild.getPath(), LocalFileType.OUTPUT_FILE, fileMetadata),
            new LocalFile(dirChild.getPath(), LocalFileType.OUTPUT_DIRECTORY, dirMetadata));
  }

  @Test
  public void testReferencedUnresolvedSymlink() throws Exception {
    scratch.file(
        "defs.bzl",
        """
        def _impl(ctx):
            s = ctx.actions.declare_symlink(ctx.label.name)
            ctx.actions.symlink(output = s, target_path = "does not matter")
            return DefaultInfo(files = depset([s]))

        sym = rule(_impl)
        """);
    scratch.file(
        "BUILD",
        "load(':defs.bzl', 'sym')",
        "sym(name = 'sym')",
        "filegroup(name = 'files', srcs = ['sym'])");
    ConfiguredTargetAndData ctAndData = getCtAndData("//:files");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    Artifact artifact = Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    artifact.getPath().getParentDirectory().createDirectoryAndParents();
    artifact.getPath().createSymbolicLink(fileSystem.getPath("/some/path"));
    FileArtifactValue metadata = FileArtifactValue.createForUnresolvedSymlink(artifact.getPath());
    CompletionContext completionContext =
        getCompletionContext(ImmutableMap.of(artifact, metadata), ImmutableMap.of());

    TargetCompleteEvent event =
        TargetCompleteEvent.successfulBuild(
            ctAndData,
            completionContext,
            artifactsToBuild.getAllArtifactsByOutputGroup(),
            /* announceTargetSummary= */ false);

    assertThat(event.referencedLocalFiles())
        .containsExactly(new LocalFile(artifact.getPath(), LocalFileType.OUTPUT_SYMLINK, metadata));
  }

  /** Regression test for b/165671166. */
  @Test
  public void testFileProtoFromArtifactReencodesAsUtf8() throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      // Windows filesystems return paths with wide characters and don't suffer from the current
      // workaround where arbitrary bytes are represented to Java as Latin-1.
      return;
    }
    scratch.file("sh/BUILD", "filegroup(name = 'globby', srcs = glob(['dir/*']))");
    // Bytes are UTF-8 encoding of: sh/dir/圖片
    byte[] filenameBytes = {
      0x73, 0x68, 0x2f, 0x64, 0x69, 0x72, 0x2f, -27, -100, -106, -25, -119, -121
    };
    String utf8InLatin1FileName = new String(filenameBytes, ISO_8859_1);
    scratch.file(utf8InLatin1FileName, "content does not matter");
    ConfiguredTargetAndData ctAndData = getCtAndData("//sh:globby");
    ArtifactsToBuild artifactsToBuild = getArtifactsToBuild(ctAndData);
    Artifact artifact = Iterables.getOnlyElement(artifactsToBuild.getAllArtifacts().toList());
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);

    File fileProto = TargetCompleteEvent.newFile(artifact, metadata);

    // Bytes are the same but the encoding is actually UTF-8 as required of a protobuf string.
    assertThat(fileProto.getName()).isEqualTo(new String(filenameBytes, UTF_8));
  }

  @Test
  public void testFileProtoFromExternalSourceArtifact() throws Exception {
    ArtifactRoot externalRoot =
        ArtifactRoot.asExternalSourceRoot(Root.fromPath(scratch.dir("/output_base/external/foo")));
    Artifact artifact =
        ActionsTestUtil.createArtifactWithExecPath(
            externalRoot, PathFragment.create("external/foo/bar/baz.cc"));
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);

    File fileProto = TargetCompleteEvent.newFile(artifact, metadata);

    assertThat(fileProto.getName()).isEqualTo("external/foo/bar/baz.cc");
    assertThat(fileProto.getPathPrefixList()).isEmpty();
  }

  @Test
  public void testFileProtoFromExternalSourceArtifact_siblingRepositoryLayout() throws Exception {
    ArtifactRoot externalRoot =
        ArtifactRoot.asExternalSourceRoot(Root.fromPath(scratch.dir("/output_base/external/foo")));
    Artifact artifact =
        ActionsTestUtil.createArtifactWithExecPath(
            externalRoot, PathFragment.create("../foo/bar/baz.cc"));
    FileArtifactValue metadata =
        FileArtifactValue.createForNormalFile(new byte[] {1, 2, 3}, null, 10);

    File fileProto = TargetCompleteEvent.newFile(artifact, metadata);

    assertThat(fileProto.getName()).isEqualTo("../foo/bar/baz.cc");
    assertThat(fileProto.getPathPrefixList()).isEmpty();
  }

  private ConfiguredTargetAndData getCtAndData(String target) throws Exception {
    AnalysisResult result = update(target);
    ConfiguredTarget ct = Iterables.getOnlyElement(result.getTargetsToBuild());
    TargetAndConfiguration tac = Iterables.getOnlyElement(result.getTopLevelTargetsWithConfigs());
    var configuredTargetConfiguration =
        (BuildConfigurationValue)
            skyframeExecutor.getEvaluator().getExistingValue(ct.getConfigurationKey());
    return new ConfiguredTargetAndData(ct, tac.getTarget(), configuredTargetConfiguration, null);
  }

  private static ArtifactsToBuild getArtifactsToBuild(ConfiguredTargetAndData ctAndData) {
    return getArtifactsToBuild(ctAndData, OutputGroupInfo.DEFAULT_GROUPS);
  }

  private static ArtifactsToBuild getArtifactsToBuild(
      ConfiguredTargetAndData ctAndData, ImmutableSortedSet<String> outputGroups) {
    TopLevelArtifactContext context =
        new TopLevelArtifactContext(
            false,
            outputGroups,
            /* failOnUnknownOutputGroups= */ false,
            /* forRunCommand= */ false);
    return TopLevelArtifactHelper.getAllArtifactsToBuild(ctAndData.getConfiguredTarget(), context);
  }

  private static CompletionContext getCompletionContext(
      Map<Artifact, FileArtifactValue> metadata,
      Map<TreeArtifact, TreeArtifactValue> treeMetadata) {
    ActionInputMap inputMap = new ActionInputMap(0);
    metadata.forEach(inputMap::put);
    treeMetadata.forEach(inputMap::putTreeArtifact);
    return new CompletionContext(ArtifactPathResolver.IDENTITY, inputMap);
  }
}
