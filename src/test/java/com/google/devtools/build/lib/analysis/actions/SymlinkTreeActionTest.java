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
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.util.ActionTester;
import com.google.devtools.build.lib.analysis.util.ActionTester.ActionCombinationFactory;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link SymlinkTreeAction}.
 */
@RunWith(JUnit4.class)
public class SymlinkTreeActionTest extends BuildViewTestCase {
  private enum KeyAttributes {
    FILESET,
    RUNFILES_FLAG,
    INPROCESS,
    FIXED_ENVIRONMENT,
    VARIABLE_ENVIRONMENT
  }

  @Test
  public void testComputeKey() throws Exception {
    final Artifact inputManifest = getBinArtifactWithNoOwner("dir/manifest.in");
    final Artifact outputManifest = getBinArtifactWithNoOwner("dir/MANIFEST");
    final Artifact runfile = getBinArtifactWithNoOwner("dir/runfile");

    ActionTester.runTest(
        KeyAttributes.class,
        new ActionCombinationFactory<KeyAttributes>() {
          @Override
          public Action generate(ImmutableSet<KeyAttributes> attributesToFlip) {
            boolean filesetTree = attributesToFlip.contains(KeyAttributes.FILESET);
            boolean enableRunfiles = attributesToFlip.contains(KeyAttributes.RUNFILES_FLAG);
            boolean inprocessSymlinkCreation = attributesToFlip.contains(KeyAttributes.INPROCESS);

            ActionEnvironment env =
                ActionEnvironment.create(
                    attributesToFlip.contains(KeyAttributes.FIXED_ENVIRONMENT)
                        ? ImmutableMap.of("a", "b")
                        : ImmutableMap.of(),
                    attributesToFlip.contains(KeyAttributes.VARIABLE_ENVIRONMENT)
                        ? ImmutableSet.of("c")
                        : ImmutableSet.of());

            // SymlinkTreeAction currently requires that (runfiles == null) == filsetTree, which the
            // ActionTester doesn't support. We therefore can't check that two actions have
            // different fingerprints when they have different runfiles objects.
            Runfiles runfiles =
                attributesToFlip.contains(KeyAttributes.FILESET)
                    ? null
                    : new Runfiles.Builder("TESTING", false).addArtifact(runfile).build();

            return new SymlinkTreeAction(
                ActionsTestUtil.NULL_ACTION_OWNER,
                inputManifest,
                runfiles,
                outputManifest,
                filesetTree,
                env,
                enableRunfiles,
                inprocessSymlinkCreation);
          }
        },
        actionKeyContext);
  }
}
