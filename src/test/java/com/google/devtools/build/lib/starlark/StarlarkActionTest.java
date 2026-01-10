// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.NULL_ACTION_OWNER;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.ensureMemoizedIsInitializedIsSet;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.dumpStructureWithEquivalenceReduction;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.InputDiscoveringNullAction;
import com.google.devtools.build.lib.analysis.actions.StarlarkAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.serialization.ArrayCodec;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationDepsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class StarlarkActionTest extends BuildViewTestCase {

  @Test
  public void serializationRoundTrip_resetsInputs() throws Exception {
    PathFragment executable = scratch.file("/bin/xxx").asFragment();
    ArtifactRoot src = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/src")));
    Artifact discoveredInput =
        ActionsTestUtil.createArtifact(src, scratch.file("/src/discovered.in"));
    Artifact.DerivedArtifact output = getBinArtifactWithNoOwner("output");
    output.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);

    StarlarkAction starlarkAction =
        (StarlarkAction)
            new StarlarkAction.Builder()
                .setShadowedAction(Optional.of(new InputDiscoveringNullAction()))
                .setExecutable(executable)
                .addOutput(output)
                .build(NULL_ACTION_OWNER, targetConfig);

    ensureMemoizedIsInitializedIsSet(starlarkAction);
    String originalStructure = dumpStructureWithEquivalenceReduction(starlarkAction);

    starlarkAction.updateDiscoveredInputs(NestedSetBuilder.create(Order.STABLE_ORDER, discoveredInput));

    new SerializationTester(starlarkAction)
        .makeMemoizingAndAllowFutureBlocking(/* allowFutureBlocking= */ true)
        .addCodec(ArrayCodec.forComponentType(Artifact.class))
        .setVerificationFunction(
            (unusedInput, deserialized) ->
                assertThat(dumpStructureWithEquivalenceReduction(deserialized))
                    .isEqualTo(originalStructure))
        .addDependencies(getCommonSerializationDependencies())
        .addDependencies(SerializationDepsUtils.SERIALIZATION_DEPS_FOR_TEST)
        .runTests();
  }
}
