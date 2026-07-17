// Copyright 2023 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.util.SkyframeIntegrationTestBase;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.vfs.DelegatingSyscallCache;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Skyframe integration tests where the {@link SyscallCache} throws {@link IOException}s. */
@RunWith(TestParameterInjector.class)
public class SkyframeFilesystemIntegrationTest extends SkyframeIntegrationTestBase {
  private final SyscallCache syscallCache = spy(SyscallCache.NO_CACHE);

  @Override
  protected ImmutableSet<EventKind> additionalEventsToCollect() {
    return ImmutableSet.of(EventKind.FINISH);
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    DelegatingSyscallCache delegatingSyscallCache = new DelegatingSyscallCache();
    delegatingSyscallCache.setDelegate(syscallCache);
    return super.getRuntimeBuilder()
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public void workspaceInit(
                  BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
                builder.setSyscallCache(delegatingSyscallCache);
              }
            });
  }

  @Test
  public void noChangeSourceFile_dirtinessCheckerIoException_noActionExecution() throws Exception {
    write(
        "hello/BUILD",
        "genrule(name='target', srcs = ['one'], outs=['out1'], cmd='/bin/cat $(SRCS) > $@')");
    Path pathToThrow = write("hello/one", "original lines");

    buildTarget("//hello:target");

    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //hello:target");

    // Throw an IOException during the initial filesystem access of the source file dep for
    // configured target //hello:target.
    initializeFileSystemSingletonIoException(pathToThrow);

    buildTarget("//hello:target");

    verify(syscallCache, atLeastOnce()).statIfFound(pathToThrow, Symlinks.NOFOLLOW);
    // Configured target //hello:target was dirtied during SkyValue dirtiness checking of
    // FileStateValue hello/one due to the thrown IOException. During the execution phase,
    // //hello:target is marked clean and not executed because hello/one is re-computed to the same
    // value with a subsequent successful filesystem access.
    MoreAsserts.assertDoesNotContainEvent(events.collector(), "Executing genrule //hello:target");
  }

  @Test
  public void changeSourceFile_dirtinessCheckerIoException_actionExecution() throws Exception {
    write(
        "hello/BUILD",
        "genrule(name='target', srcs = ['one'], outs=['out1'], cmd='/bin/cat $(SRCS) > $@')");
    write("hello/one", "original lines");

    buildTarget("//hello:target");

    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //hello:target");

    // Throw an IOException during the initial filesystem access of the source file dep for
    // configured target //hello:target.
    Path pathToThrow = write("hello/one", "new lines");
    initializeFileSystemSingletonIoException(pathToThrow);

    buildTarget("//hello:target");

    verify(syscallCache, atLeastOnce()).statIfFound(pathToThrow, Symlinks.NOFOLLOW);
    // Configured target //hello:target was dirtied during SkyValue dirtiness checking of
    // FileStateValue hello/one due to the thrown IOException. During the execution phase,
    // //hello:target is re-executed because hello/one re-computed to a different value with a
    // subsequent successful filesystem access.
    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //hello:target");
  }

  @Test
  public void persistentIoException_inputFileError() throws Exception {
    write(
        "hello/BUILD",
        "genrule(name='target', srcs = ['one'], outs=['out1'], cmd='/bin/cat $(SRCS) > $@')");
    Path pathToThrow = write("hello/one", "original lines");

    buildTarget("//hello:target");

    MoreAsserts.assertContainsEvent(events.collector(), "Executing genrule //hello:target");

    initializeFileSystemPersistentIoException(pathToThrow);

    BuildFailedException e =
        assertThrows(BuildFailedException.class, () -> buildTarget("//hello:target"));

    verify(syscallCache, atLeastOnce()).statIfFound(pathToThrow, Symlinks.NOFOLLOW);
    assertThat(e.getDetailedExitCode().getFailureDetail().getExecution().getCode())
        .isEqualTo(Code.SOURCE_INPUT_IO_EXCEPTION);
    assertThat(e.getDetailedExitCode().getFailureDetail().getMessage())
        .contains("error reading file '//hello:one'");
  }

  private void initializeFileSystemSingletonIoException(Path pathToThrow) throws IOException {
    when(syscallCache.statIfFound(pathToThrow, Symlinks.NOFOLLOW))
        .thenThrow(new IOException("filesystem error"))
        .thenReturn(SyscallCache.NO_CACHE.statIfFound(pathToThrow, Symlinks.NOFOLLOW));
  }

  private void initializeFileSystemPersistentIoException(Path pathToThrow) throws IOException {
    when(syscallCache.statIfFound(pathToThrow, Symlinks.NOFOLLOW))
        .thenThrow(new IOException("filesystem error"));
  }
}
