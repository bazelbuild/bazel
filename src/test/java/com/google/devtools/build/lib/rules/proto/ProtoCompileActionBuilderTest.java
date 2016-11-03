// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.createCommandLineFromToolchains;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProtoCompileActionBuilderTest {

  private final Root root = Root.asSourceRoot(new InMemoryFileSystem().getPath("/"));

  @Test
  public void commandLine_basic() throws Exception {
    FilesToRunProvider plugin =
        new FilesToRunProvider(
            ImmutableList.<Artifact>of(),
            null /* runfilesSupport */,
            artifact("protoc-gen-javalite.exe"));

    ProtoLangToolchainProvider toolchainNoPlugin =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:$(OUT)",
            null /* pluginExecutable */,
            mock(TransitiveInfoCollection.class) /* runtime */,
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER) /* blacklistedProtos */);

    ProtoLangToolchainProvider toolchainWithPlugin =
        ProtoLangToolchainProvider.create(
            "--$(PLUGIN_OUT)=param3,param4:$(OUT)",
            plugin,
            mock(TransitiveInfoCollection.class) /* runtime */,
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER) /* blacklistedProtos */);

    SupportData supportData =
        SupportData.create(
            Predicates.<TransitiveInfoCollection>alwaysFalse(),
            ImmutableList.of(artifact("source_file.proto")),
            NestedSetBuilder.create(
                STABLE_ORDER, artifact("import1.proto"), artifact("import2.proto")),
            null /* usedDirectDeps */,
            true /* hasProtoSources */);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(
                new ToolchainInvocation(
                    "dontcare_because_no_plugin", toolchainNoPlugin, "foo.srcjar"),
                new ToolchainInvocation("pluginName", toolchainWithPlugin, "bar.srcjar")),
            supportData,
            true /* allowServices */,
            ImmutableList.<String>of() /* protocOpts */);

    assertThat(cmdLine.arguments())
        .containsExactly(
            "--java_out=param1,param2:foo.srcjar",
            "--PLUGIN_pluginName_out=param3,param4:bar.srcjar",
            "--plugin=protoc-gen-PLUGIN_pluginName=protoc-gen-javalite.exe",
            "-Iimport1.proto=import1.proto",
            "-Iimport2.proto=import2.proto",
            "source_file.proto")
        .inOrder();
  }

  @Test
  public void otherParameters() throws Exception {
    SupportData supportData =
        SupportData.create(
            Predicates.<TransitiveInfoCollection>alwaysFalse(),
            ImmutableList.<Artifact>of(),
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER),
            null /* usedDirectDeps */,
            true /* hasProtoSources */);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.<ToolchainInvocation>of(),
            supportData,
            false /* allowServices */,
            ImmutableList.of("--foo", "--bar") /* protocOpts */);

    assertThat(cmdLine.arguments()).containsAllOf("--disallow_services", "--foo", "--bar");
  }

  @Test
  public void outReplacementAreLazilyEvaluated() throws Exception {
    final boolean[] hasBeenCalled = new boolean[1];
    hasBeenCalled[0] = false;

    CharSequence outReplacement =
        new LazyString() {
          @Override
          public String toString() {
            hasBeenCalled[0] = true;
            return "mu";
          }
        };

    ProtoLangToolchainProvider toolchain =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:$(OUT)",
            null /* pluginExecutable */,
            mock(TransitiveInfoCollection.class) /* runtime */,
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER) /* blacklistedProtos */);

    SupportData supportData =
        SupportData.create(
            Predicates.<TransitiveInfoCollection>alwaysFalse(),
            ImmutableList.<Artifact>of(),
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER),
            null /* usedDirectDeps */,
            true /* hasProtoSources */);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(new ToolchainInvocation("pluginName", toolchain, outReplacement)),
            supportData,
            true /* allowServices */,
            ImmutableList.<String>of() /* protocOpts */);

    assertThat(hasBeenCalled[0]).isFalse();
    cmdLine.arguments();
    assertThat(hasBeenCalled[0]).isTrue();
  }

  /**
   * Tests that if the same invocation-name is specified by more than one invocation,
   * ProtoCompileActionBuilder throws an exception.
   */
  @Test
  public void exceptionIfSameName() throws Exception {
    SupportData supportData =
        SupportData.create(
            Predicates.<TransitiveInfoCollection>alwaysFalse(),
            ImmutableList.<Artifact>of(),
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER),
            null /* usedDirectDeps */,
            true /* hasProtoSources */);

    ProtoLangToolchainProvider toolchain1 =
        ProtoLangToolchainProvider.create(
            "dontcare",
            null /* pluginExecutable */,
            mock(TransitiveInfoCollection.class) /* runtime */,
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER) /* blacklistedProtos */);

    ProtoLangToolchainProvider toolchain2 =
        ProtoLangToolchainProvider.create(
            "dontcare",
            null /* pluginExecutable */,
            mock(TransitiveInfoCollection.class) /* runtime */,
            NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER) /* blacklistedProtos */);

    try {
      createCommandLineFromToolchains(
          ImmutableList.of(
              new ToolchainInvocation("pluginName", toolchain1, "outReplacement"),
              new ToolchainInvocation("pluginName", toolchain2, "outReplacement")),
          supportData,
          true /* allowServices */,
          ImmutableList.<String>of() /* protocOpts */);
      fail("Expected an exception");
    } catch (IllegalStateException e) {
      assertThat(e.getMessage())
          .isEqualTo(
              "Invocation name pluginName appears more than once. "
                  + "This could lead to incorrect proto-compiler behavior");
    }
  }

  private Artifact artifact(String path) {
    return new Artifact(new PathFragment(path), root);
  }
}
