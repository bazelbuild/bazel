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
package com.google.devtools.build.lib.analysis.actions;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link SymlinkTreeAction}. */
@RunWith(JUnit4.class)
public class SymlinkTreeActionTest extends BuildViewTestCase {
  private enum FilesetActionAttributes {
    ENABLE_RUNFILES,
    INPROCESS_SYMLINKS,
    FIXED_ENVIRONMENT,
    VARIABLE_ENVIRONMENT
  }

  private enum RunfilesActionAttributes {
    RUNFILES,
    ENABLE_RUNFILES,
    INPROCESS_SYMLINKS,
    FIXED_ENVIRONMENT,
    VARIABLE_ENVIRONMENT
  }

  private enum SkipManifestAttributes {
    RUNFILES,
    INPROCESS_SYMLINKS,
    FIXED_ENVIRONMENT,
    VARIABLE_ENVIRONMENT
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact inputManifest = getBinArtifactWithNoOwner("dir/manifest.in");
    final Artifact outputManifest = getBinArtifactWithNoOwner("dir/MANIFEST");
    final Artifact runfile = getBinArtifactWithNoOwner("dir/runfile");
    final Artifact runfile2 = getBinArtifactWithNoOwner("dir/runfile2");

    new ActionTester(actionKeyContext)
        .combinations(
            RunfilesActionAttributes.class,
            (attributesToFlip) ->
                new SymlinkTreeAction(
                    ActionsTestUtil.NULL_ACTION_OWNER,
                    inputManifest,
                    /*runfiles=*/ attributesToFlip.contains(RunfilesActionAttributes.RUNFILES)
                        ? new Runfiles.Builder("TESTING", false).addArtifact(runfile).build()
                        : new Runfiles.Builder("TESTING", false).addArtifact(runfile2).build(),
                    outputManifest,
                    /*filesetTree=*/ false,
                    createActionEnvironment(
                        attributesToFlip.contains(RunfilesActionAttributes.FIXED_ENVIRONMENT),
                        attributesToFlip.contains(RunfilesActionAttributes.VARIABLE_ENVIRONMENT)),
                    attributesToFlip.contains(RunfilesActionAttributes.ENABLE_RUNFILES),
                    attributesToFlip.contains(RunfilesActionAttributes.INPROCESS_SYMLINKS),
                    /*skipRunfilesManifests=*/ false))
        .combinations(
            FilesetActionAttributes.class,
            (attributesToFlip) ->
                new SymlinkTreeAction(
                    ActionsTestUtil.NULL_ACTION_OWNER,
                    inputManifest,
                    /*runfiles=*/ null,
                    outputManifest,
                    /*filesetTree=*/ true,
                    createActionEnvironment(
                        attributesToFlip.contains(FilesetActionAttributes.FIXED_ENVIRONMENT),
                        attributesToFlip.contains(FilesetActionAttributes.VARIABLE_ENVIRONMENT)),
                    attributesToFlip.contains(FilesetActionAttributes.ENABLE_RUNFILES),
                    attributesToFlip.contains(FilesetActionAttributes.INPROCESS_SYMLINKS),
                    /*skipRunfilesManifests=*/ false))
        .combinations(
            SkipManifestAttributes.class,
            (attributesToFlip) ->
                // skipRunfilesManifests requires !filesetTree and enableRunfiles
                new SymlinkTreeAction(
                    ActionsTestUtil.NULL_ACTION_OWNER,
                    inputManifest,
                    attributesToFlip.contains(SkipManifestAttributes.RUNFILES)
                        ? new Runfiles.Builder("TESTING", false).addArtifact(runfile).build()
                        : new Runfiles.Builder("TESTING", false).addArtifact(runfile2).build(),
                    outputManifest,
                    /*filesetTree=*/ false,
                    createActionEnvironment(
                        attributesToFlip.contains(SkipManifestAttributes.FIXED_ENVIRONMENT),
                        attributesToFlip.contains(SkipManifestAttributes.VARIABLE_ENVIRONMENT)),
                    /*enableRunfiles=*/ true,
                    attributesToFlip.contains(SkipManifestAttributes.INPROCESS_SYMLINKS),
                    /*skipRunfilesManifests=*/ true))
        .runTest();
  }

  private static ActionEnvironment createActionEnvironment(boolean fixed, boolean variable) {
    return ActionEnvironment.create(
        fixed ? ImmutableMap.of("a", "b") : ImmutableMap.of(),
        variable ? ImmutableSet.of("c") : ImmutableSet.of());
  }

  @Test
  public void testNullRunfilesThrows() {
    Artifact inputManifest = getBinArtifactWithNoOwner("dir/manifest.in");
    Artifact outputManifest = getBinArtifactWithNoOwner("dir/MANIFEST");
    MoreAsserts.assertThrows(
        IllegalArgumentException.class,
        () ->
            new SymlinkTreeAction(
                ActionsTestUtil.NULL_ACTION_OWNER,
                inputManifest,
                /*runfiles=*/ null,
                outputManifest,
                /*filesetTree=*/ false,
                createActionEnvironment(false, false),
                /*enableRunfiles=*/ true,
                /*inprocessSymlinkCreation=*/ false,
                /*skipRunfilesManifests=*/ false));
  }
}
