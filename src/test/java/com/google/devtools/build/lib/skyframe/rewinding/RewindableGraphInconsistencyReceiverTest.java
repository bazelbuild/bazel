// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.rewinding;

import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RewindableGraphInconsistencyReceiver}. */
@RunWith(JUnit4.class)
public final class RewindableGraphInconsistencyReceiverTest {

  /** A key of a given function whose type is all that the receiver looks at. */
  private record TestKey(SkyFunctionName functionName, String name) implements SkyKey {}

  private static final SkyKey ACTION =
      ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, /* actionIndex= */ 0);
  private static final SkyKey REPO_FETCH = new TestKey(SkyFunctions.REPOSITORY_DIRECTORY, "repo");
  private static final SkyKey FILE_STATE = new TestKey(FileStateKey.FILE_STATE, "file");
  // Reads REPO.bazel of a repo and may thus find its fetch undone after a restart, but neither
  // rewinds nor is rewound itself.
  private static final SkyKey REPO_FILE = new TestKey(SkyFunctions.REPO_FILE, "repo");

  private final RewindableGraphInconsistencyReceiver receiver =
      new RewindableGraphInconsistencyReceiver(
          /* heuristicallyDropNodes= */ false, /* skymeldInconsistenciesExpected= */ false);

  @Test
  public void resetRequested_parentOfUndoneLostSourceChild_accepted() {
    initiateRewinding();

    receiver.noteInconsistencyAndMaybeThrow(
        REPO_FILE, ImmutableList.of(REPO_FETCH), Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);
    // The evaluator resets a parent that had already requested the undone child.
    receiver.noteInconsistencyAndMaybeThrow(
        REPO_FILE, /* otherKeys= */ null, Inconsistency.RESET_REQUESTED);
  }

  @Test
  public void resetRequested_parentOfUndoneLostSourceChild_acceptedOnce() {
    initiateRewinding();
    receiver.noteInconsistencyAndMaybeThrow(
        REPO_FILE, ImmutableList.of(FILE_STATE), Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);
    receiver.noteInconsistencyAndMaybeThrow(
        REPO_FILE, /* otherKeys= */ null, Inconsistency.RESET_REQUESTED);

    assertThrows(
        IllegalStateException.class,
        () ->
            receiver.noteInconsistencyAndMaybeThrow(
                REPO_FILE, /* otherKeys= */ null, Inconsistency.RESET_REQUESTED));
  }

  @Test
  public void resetRequested_typeThatNeverResets_rejected() {
    initiateRewinding();

    assertThrows(
        IllegalStateException.class,
        () ->
            receiver.noteInconsistencyAndMaybeThrow(
                REPO_FILE, /* otherKeys= */ null, Inconsistency.RESET_REQUESTED));
  }

  @Test
  public void reset_forgetsParentsOfUndoneLostSourceChildren() {
    initiateRewinding();
    receiver.noteInconsistencyAndMaybeThrow(
        REPO_FILE, ImmutableList.of(REPO_FETCH), Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);

    receiver.reset();
    initiateRewinding();

    assertThrows(
        IllegalStateException.class,
        () ->
            receiver.noteInconsistencyAndMaybeThrow(
                REPO_FILE, /* otherKeys= */ null, Inconsistency.RESET_REQUESTED));
  }

  /** Rewinds the file state of a lost source file from an action, as a lost input does. */
  private void initiateRewinding() {
    receiver.noteInconsistencyAndMaybeThrow(
        ACTION, ImmutableList.of(FILE_STATE), Inconsistency.PARENT_FORCE_REBUILD_OF_CHILD);
  }
}
