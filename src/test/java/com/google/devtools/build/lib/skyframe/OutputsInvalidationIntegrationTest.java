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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Integration test for action invalidation based on output modifications returned by {@link
 * OutputService#startBuild}.
 */
@RunWith(TestParameterInjector.class)
public final class OutputsInvalidationIntegrationTest extends BuildIntegrationTestCase {

  private final OutputService outputService = mock(OutputService.class);

  @Before
  public void prepareOutputServiceMock()
      throws BuildFailedException, AbruptExitException, InterruptedException, IOException {
    when(outputService.actionFileSystemType()).thenReturn(ActionFileSystemType.DISABLED);
    when(outputService.getFilesSystemName()).thenReturn("fileSystemName");
    when(outputService.startBuild(any(), any(), anyBoolean()))
        .thenReturn(ModifiedFileSet.EVERYTHING_MODIFIED);
    when(outputService.getXattrProvider(any())).thenAnswer(i -> i.getArgument(0));
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public OutputService getOutputService() {
                return outputService;
              }
            });
  }

  @Override
  protected EventCollectionApparatus createEvents() {
    return new EventCollectionApparatus(ImmutableSet.of(EventKind.FINISH));
  }

  @Test
  public void nothingModified_doesntInvalidateAnyActions(@TestParameter boolean deleteOutput)
      throws Exception {
    write("foo/BUILD", "genrule(name='foo', outs=['foo.out'], cmd='touch $@')");
    buildTarget("//foo");
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");
    if (deleteOutput) {
      delete(getOnlyOutput("//foo"));
    }

    when(outputService.startBuild(any(), any(), anyBoolean()))
        .thenReturn(ModifiedFileSet.NOTHING_MODIFIED);
    events.collector().clear();
    buildTarget("//foo");

    MoreAsserts.assertDoesNotContainEvent(events.collector(), "Executing genrule //foo:foo");
  }

  private enum ReportedModification {
    EVERYTHING_MODIFIED {
      @Override
      ModifiedFileSet modifiedFileSet(Artifact artifact) {
        return ModifiedFileSet.EVERYTHING_MODIFIED;
      }
    },
    EVERYTHING_DELETED {
      @Override
      ModifiedFileSet modifiedFileSet(Artifact artifact) {
        return ModifiedFileSet.EVERYTHING_DELETED;
      }
    },
    SINGLE_FILE {
      @Override
      ModifiedFileSet modifiedFileSet(Artifact artifact) {
        return ModifiedFileSet.builder().modify(artifact.getExecPath()).build();
      }
    };

    abstract ModifiedFileSet modifiedFileSet(Artifact artifact);
  }

  @Test
  public void identicalOutputs_doesntInvalidateAnyActions(
      @TestParameter ReportedModification modification) throws Exception {
    write("foo/BUILD", "genrule(name='foo', outs=['foo.out'], cmd='touch $@')");
    buildTarget("//foo");
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");

    when(outputService.startBuild(any(), any(), anyBoolean()))
        .thenReturn(modification.modifiedFileSet(getOnlyOutput("//foo")));
    events.collector().clear();
    buildTarget("//foo");

    MoreAsserts.assertDoesNotContainEvent(events.collector(), "Executing genrule //foo:foo");
  }

  @Test
  public void noCheckOutputFiles_ignoresModifiedFiles(
      @TestParameter ReportedModification modification) throws Exception {
    addOptions("--experimental_check_output_files");
    write("foo/BUILD", "genrule(name='foo', outs=['foo.out'], cmd='touch $@')");
    buildTarget("//foo");
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");

    when(outputService.startBuild(any(), any(), anyBoolean()))
        .thenReturn(modification.modifiedFileSet(getOnlyOutput("//foo")));
    events.collector().clear();
    buildTarget("//foo");

    MoreAsserts.assertDoesNotContainEvent(events.collector(), "Executing genrule //foo:foo");
  }

  @TestParameters({
    "{everythingDeleted: false, checkOutputFiles: true}",
    "{everythingDeleted: true, checkOutputFiles: false}",
    "{everythingDeleted: true, checkOutputFiles: true}",
  })
  @Test
  public void everythingModified_invalidatesAllActions(
      boolean everythingDeleted, boolean checkOutputFiles) throws Exception {
    addOptions("--experimental_check_output_files=" + checkOutputFiles);
    write("foo/BUILD", "genrule(name='foo', outs=['foo.out'], cmd='touch $@')");
    buildTarget("//foo");
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");
    delete(getOnlyOutput("//foo"));

    when(outputService.startBuild(any(), any(), anyBoolean()))
        .thenReturn(
            everythingDeleted
                ? ModifiedFileSet.EVERYTHING_DELETED
                : ModifiedFileSet.EVERYTHING_MODIFIED);
    events.collector().clear();
    buildTarget("//foo");

    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");
  }

  @Test
  public void outputFileModified_invalidatesOnlyAffectedAction() throws Exception {
    write(
        "foo/BUILD",
        "genrule(name='foo', outs=['foo.out'], cmd='touch $@')",
        "genrule(name='bar', outs=['bar.out'], cmd='touch $@')");
    buildTarget("//foo:all");
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:bar");
    Artifact fooOut = getOnlyOutput("//foo");
    delete(fooOut);

    when(outputService.startBuild(any(), any(), anyBoolean())).thenReturn(modifiedFileSet(fooOut));
    events.collector().clear();
    buildTarget("//foo:all");

    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //foo:foo");
    MoreAsserts.assertDoesNotContainEvent(events.collector(), "Executing genrule //foo:bar");
  }

  private static void delete(Artifact artifact) throws IOException {
    assertThat(artifact.getPath().delete()).isTrue();
  }

  private Artifact getOnlyOutput(String label) throws Exception {
    return getConfiguredTarget(label)
        .getProvider(FileProvider.class)
        .getFilesToBuild()
        .getSingleton();
  }

  private ModifiedFileSet modifiedFileSet(Artifact... artifacts) {
    ModifiedFileSet.Builder modifiedFileSet = ModifiedFileSet.builder();
    for (Artifact artifact : artifacts) {
      modifiedFileSet.modify(artifact.getExecPath());
    }
    return modifiedFileSet.build();
  }
}
