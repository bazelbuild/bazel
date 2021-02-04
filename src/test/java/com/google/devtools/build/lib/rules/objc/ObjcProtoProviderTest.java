// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ObjcProvider}. */
@RunWith(JUnit4.class)
public class ObjcProtoProviderTest extends ObjcRuleTestCase {

  @Before
  public final void setup() throws Exception {
    // Empty target just so artifacts can be created.
    scratch.file("x/BUILD",
        "objc_library(",
        "    name = 'x',",
        ")");
  }

  @Test
  public void emptyProvider() {
    ObjcProtoProvider empty = new ObjcProtoProvider.Builder().build();
    assertThat(empty.getProtoFiles().toList()).isEmpty();
  }

  @Test
  public void onlyPropagatesProvider() throws Exception {
    Artifact foo = getTestArtifact("foo");
    ObjcProtoProvider onlyPropagates =
        new ObjcProtoProvider.Builder()
            .addProtoFiles(NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER, foo))
            .build();
    assertThat(Iterables.concat(onlyPropagates.getProtoFiles().toList())).containsExactly(foo);
  }

  @Test
  public void propagatesThroughDependers() throws Exception {
    Artifact foo = getTestArtifact("foo");
    Artifact bar = getTestArtifact("bar");
    Artifact baz = getTestArtifact("baz");
    Artifact header = getTestArtifact("protobuf");
    PathFragment searchPath = header.getExecPath().getParentDirectory();

    ObjcProtoProvider base1 =
        new ObjcProtoProvider.Builder()
            .addProtoFiles(NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER, foo))
            .addPortableProtoFilters(NestedSetBuilder.<Artifact>create(Order.STABLE_ORDER, baz))
            .addProtobufHeaders(NestedSetBuilder.<Artifact>create(Order.STABLE_ORDER, header))
            .build();

    ObjcProtoProvider base2 =
        new ObjcProtoProvider.Builder()
            .addProtoFiles(NestedSetBuilder.<Artifact>create(Order.NAIVE_LINK_ORDER, bar))
            .addProtobufHeaderSearchPaths(
                NestedSetBuilder.<PathFragment>create(Order.LINK_ORDER, searchPath))
            .build();

    ObjcProtoProvider depender =
        new ObjcProtoProvider.Builder().addTransitive(ImmutableList.of(base1, base2)).build();
    assertThat(Iterables.concat(depender.getProtoFiles().toList())).containsExactly(foo, bar);
    assertThat(depender.getPortableProtoFilters().toList()).containsExactly(baz);
    assertThat(depender.getProtobufHeaders().toList()).containsExactly(header);
    assertThat(depender.getProtobufHeaderSearchPaths().toList()).containsExactly(searchPath);
  }

  private Artifact getTestArtifact(String name) throws Exception {
    ConfiguredTarget target = getConfiguredTarget("//x:x");
    return getBinArtifact(name, target);
  }
}
