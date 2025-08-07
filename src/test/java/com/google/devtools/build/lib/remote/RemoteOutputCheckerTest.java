package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteOutputChecker} */
@RunWith(JUnit4.class)
public class RemoteOutputCheckerTest {
  private final RemoteOutputChecker remoteOutputChecker =
      new RemoteOutputChecker("build", RemoteOutputsMode.MINIMAL, ImmutableList.of());
  private final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final ArtifactRoot execRoot =
      ArtifactRoot.asDerivedRoot(fs.getPath("/execroot"), ArtifactRoot.RootType.OUTPUT, "out");

  @Test
  public void testShouldDownloadOutput() {
    remoteOutputChecker.addOutputToDownload(
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(execRoot, "foo/bar"));
    remoteOutputChecker.addOutputToDownload(
        ActionsTestUtil.createArtifact(execRoot, "foo/bar-baz"));
    assertThat(
            remoteOutputChecker.shouldDownloadOutput(PathFragment.create("out/foo/bar-quz"), null))
        .isFalse();
    assertThat(remoteOutputChecker.shouldDownloadOutput(PathFragment.create("out/foo/bar"), null))
        .isTrue();
    assertThat(
            remoteOutputChecker.shouldDownloadOutput(
                PathFragment.create("out/foo/bar/data.txt"), PathFragment.create("out/foo/bar")))
        .isTrue();
    assertThat(
            remoteOutputChecker.shouldDownloadOutput(PathFragment.create("out/foo/bar-baz"), null))
        .isTrue();
  }
}
