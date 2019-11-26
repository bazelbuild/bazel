// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/** Unit tests for {@link SymlinkTreeStrategy}. */
@RunWith(JUnit4.class)
public final class SymlinkTreeStrategyTest extends BuildViewTestCase {
  @Test
  public void testArtifactToPathConversion() {
    Artifact artifact = getBinArtifactWithNoOwner("dir/foo");
    assertThat(SymlinkTreeStrategy.TO_PATH.apply(artifact))
        .isEqualTo(artifact.getPath().asFragment());
    assertThat(SymlinkTreeStrategy.TO_PATH.apply(null)).isEqualTo(null);
  }

  @Test
  public void outputServiceInteraction() throws Exception {
    ActionExecutionContext context = mock(ActionExecutionContext.class);
    OutputService outputService = mock(OutputService.class);
    StoredEventHandler eventHandler = new StoredEventHandler();

    when(context.getContext(SymlinkTreeActionContext.class))
        .thenReturn(new SymlinkTreeStrategy(outputService, null));
    when(context.getInputPath(any())).thenAnswer((i) -> ((Artifact) i.getArgument(0)).getPath());
    when(context.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);
    when(context.getEventHandler()).thenReturn(eventHandler);
    when(outputService.canCreateSymlinkTree()).thenReturn(true);

    Artifact inputManifest = getBinArtifactWithNoOwner("dir/manifest.in");
    Artifact outputManifest = getBinArtifactWithNoOwner("dir/MANIFEST");
    Artifact runfile = getBinArtifactWithNoOwner("dir/runfile");
    doAnswer(
            (i) -> {
              outputManifest.getPath().getParentDirectory().createDirectoryAndParents();
              return null;
            })
        .when(outputService)
        .createSymlinkTree(any(), any());

    Runfiles runfiles =
        new Runfiles.Builder("TESTING", false)
            .setEmptyFilesSupplier((paths) -> ImmutableList.of(PathFragment.create("dir/empty")))
            .addArtifact(runfile)
            .build();
    SymlinkTreeAction action =
        new SymlinkTreeAction(
            ActionsTestUtil.NULL_ACTION_OWNER,
            inputManifest,
            runfiles,
            outputManifest,
            /*filesetTree=*/ false,
            ActionEnvironment.EMPTY,
            /*enableRunfiles=*/ true);

    action.execute(context);

    @SuppressWarnings("unchecked")
    ArgumentCaptor<Map<PathFragment, PathFragment>> capture = ArgumentCaptor.forClass(Map.class);
    verify(outputService, times(1)).createSymlinkTree(capture.capture(), any());
    assertThat(capture.getValue())
        .containsExactly(
            PathFragment.create("TESTING/dir/runfile"),
            runfile.getPath().asFragment(),
            PathFragment.create("TESTING/dir/empty"),
            null);
  }
}
