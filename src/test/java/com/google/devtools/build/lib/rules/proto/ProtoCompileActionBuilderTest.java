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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import javax.annotation.Nullable;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProtoCompileActionBuilder}. */
@RunWith(JUnit4.class)
public class ProtoCompileActionBuilderTest {

  private static final InMemoryFileSystem FILE_SYSTEM = new InMemoryFileSystem();
  private final ArtifactRoot root =
      ArtifactRoot.asSourceRoot(Root.fromPath(FILE_SYSTEM.getPath("/")));
  private final ArtifactRoot derivedRoot =
      ArtifactRoot.asDerivedRoot(FILE_SYSTEM.getPath("/"), FILE_SYSTEM.getPath("/out"));

  @Test
  public void commandLine_basic() throws Exception {
    FilesToRunProvider plugin =
        new FilesToRunProvider(
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            null /* runfilesSupport */,
            artifact("//:dont-care", "protoc-gen-javalite.exe"));

    ProtoLangToolchainProvider toolchainNoPlugin =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:$(OUT)",
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    ProtoLangToolchainProvider toolchainWithPlugin =
        ProtoLangToolchainProvider.create(
            "--$(PLUGIN_OUT)=param3,param4:$(OUT)",
            plugin,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    SupportData supportData =
        SupportData.create(
            ImmutableList.of(artifact("//:dont-care", "source_file.proto")),
            /* protosInDirectDeps= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            NestedSetBuilder.create(
                STABLE_ORDER,
                artifact("//:dont-care", "import1.proto"),
                artifact("//:dont-care", "import2.proto")),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* protoSourceRoot= */ "",
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* hasProtoSources= */ true,
            /* protosInExports= */ null,
            /* exportedProtoSourceRoots= */ null);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(
                new ToolchainInvocation(
                    "dontcare_because_no_plugin", toolchainNoPlugin, "foo.srcjar"),
                new ToolchainInvocation("pluginName", toolchainWithPlugin, "bar.srcjar")),
            supportData.getDirectProtoSources(),
            supportData.getTransitiveImports(),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.<String>stableOrder().build(),
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* exportedProtoSourceRoots= */ null,
            /* protosInDirectDeps= */ null,
            /* protosInExports= */ null,
            Label.parseAbsoluteUnchecked("//foo:bar"),
            /* allowServices= */ true,
            /* protocOpts= */ ImmutableList.of());

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
  public void commandline_derivedArtifact() {
    // Verify that the command line contains the correct path to a generated protocol buffers.
    SupportData supportData =
        SupportData.create(
            ImmutableList.of(derivedArtifact("//:dont-care", "source_file.proto")),
            /* protosInDirectDeps= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* transitiveImports= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* protoSourceRoot= */ "",
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* hasProtoSources= */ true,
            /* protosInExports= */ null,
            /* exportedProtoSourceRoots= */ null);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            /* toolchainInvocations= */ ImmutableList.of(),
            supportData.getDirectProtoSources(),
            supportData.getTransitiveImports(),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* directProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* exportedProtoSourceRoots= */ null,
            /* protosInDirectDeps= */ null,
            /* protosInExports= */ null,
            Label.parseAbsoluteUnchecked("//foo:bar"),
            /* allowServices= */ true,
            /* protocOpts= */ ImmutableList.of());

    assertThat(cmdLine.arguments()).containsExactly("out/source_file.proto");
  }

  @Test
  public void commandLine_strictDeps() throws Exception {
    ProtoLangToolchainProvider toolchain =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:$(OUT)",
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    SupportData supportData =
        SupportData.create(
            ImmutableList.of(artifact("//:dont-care", "source_file.proto")),
            NestedSetBuilder.create(STABLE_ORDER, artifact("//:dont-care", "import1.proto")),
            NestedSetBuilder.create(
                STABLE_ORDER,
                artifact("//:dont-care", "import1.proto"),
                artifact("//:dont-care", "import2.proto")),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* protoSourceRoot= */ "",
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* hasProtoSources= */ true,
            /* protosInExports= */ null,
            /* exportedProtoSourceRoots= */ null);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(new ToolchainInvocation("dontcare", toolchain, "foo.srcjar")),
            supportData.getDirectProtoSources(),
            supportData.getTransitiveImports(),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* directProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* exportedProtoSourceRoots= */ null,
            supportData.getProtosInDirectDeps(),
            /* protosInExports= */ null,
            Label.parseAbsoluteUnchecked("//foo:bar"),
            /* allowServices= */ true,
            /* protocOpts= */ ImmutableList.of());

    assertThat(cmdLine.arguments())
        .containsExactly(
            "--java_out=param1,param2:foo.srcjar",
            "-Iimport1.proto=import1.proto",
            "-Iimport2.proto=import2.proto",
            "--direct_dependencies",
            "import1.proto",
            String.format(ProtoCompileActionBuilder.STRICT_DEPS_FLAG_TEMPLATE, "//foo:bar"),
            "source_file.proto")
        .inOrder();
  }

  @Test
  public void otherParameters() throws Exception {
    SupportData supportData =
        SupportData.create(
            ImmutableList.of(),
            /* protosInDirectDeps= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            NestedSetBuilder.emptySet(STABLE_ORDER),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* protoSourceRoot= */ "",
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* hasProtoSources= */ true,
            /* protosInExports= */ null,
            /* exportedProtoSourceRoots= */ null);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(),
            supportData.getDirectProtoSources(),
            supportData.getTransitiveImports(),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* directProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* exportedProtoSourceRoots= */ null,
            supportData.getProtosInDirectDeps(),
            /* protosInExports= */ null,
            Label.parseAbsoluteUnchecked("//foo:bar"),
            /* allowServices= */ false,
            /* protocOpts= */ ImmutableList.of("--foo", "--bar"));

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
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    SupportData supportData =
        SupportData.create(
            ImmutableList.of(),
            /* protosInDirectDeps= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            NestedSetBuilder.emptySet(STABLE_ORDER),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* protoSourceRoot= */ "",
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* hasProtoSources= */ true,
            /* protosInExports= */ null,
            /* exportedProtoSourceRoots= */ null);

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(new ToolchainInvocation("pluginName", toolchain, outReplacement)),
            supportData.getDirectProtoSources(),
            supportData.getTransitiveImports(),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* directProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* exportedProtoSourceRoots= */ null,
            supportData.getProtosInDirectDeps(),
            /* protosInExports= */ null,
            Label.parseAbsoluteUnchecked("//foo:bar"),
            /* allowServices= */ true,
            /* protocOpts= */ ImmutableList.of());

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
            ImmutableList.of(),
            /* protosInDirectDeps= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            NestedSetBuilder.emptySet(STABLE_ORDER),
            /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
            /* protoSourceRoot= */ "",
            /* directProtoSourceRoots= */ NestedSetBuilder.<String>stableOrder().build(),
            /* hasProtoSources= */ true,
            /* protosInExports= */ null,
            /* exportedProtoSourceRoots= */ null);

    ProtoLangToolchainProvider toolchain1 =
        ProtoLangToolchainProvider.create(
            "dontcare",
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    ProtoLangToolchainProvider toolchain2 =
        ProtoLangToolchainProvider.create(
            "dontcare",
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    try {
      createCommandLineFromToolchains(
          ImmutableList.of(
              new ToolchainInvocation("pluginName", toolchain1, "outReplacement"),
              new ToolchainInvocation("pluginName", toolchain2, "outReplacement")),
          supportData.getDirectProtoSources(),
          supportData.getTransitiveImports(),
          /* transitiveProtoPathFlags= */ NestedSetBuilder.emptySet(STABLE_ORDER),
          /* directProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
          /* exportedProtoSourceRoots= */ null,
          supportData.getProtosInDirectDeps(),
          /* protosInExports= */ null,
          Label.parseAbsoluteUnchecked("//foo:bar"),
          /* allowServices= */ true,
          /* protocOpts= */ ImmutableList.of());
      fail("Expected an exception");
    } catch (IllegalStateException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "Invocation name pluginName appears more than once. "
                  + "This could lead to incorrect proto-compiler behavior");
    }
  }

  @Test
  public void testProtoCommandLineArgv() throws Exception {
    assertThat(
            protoArgv(
                null /* directDependencies */,
                ImmutableList.of(derivedArtifact("//:dont-care", "foo.proto"))))
        .containsExactly("-Ifoo.proto=out/foo.proto");

    assertThat(
            protoArgv(
                ImmutableList.of() /* directDependencies */,
                ImmutableList.of(derivedArtifact("//:dont-care", "foo.proto"))))
        .containsExactly("-Ifoo.proto=out/foo.proto", "--direct_dependencies=");

    assertThat(
            protoArgv(
                ImmutableList.of(
                    derivedArtifact("//:dont-care", "foo.proto")) /* directDependencies */,
                ImmutableList.of(derivedArtifact("//:dont-care", "foo.proto"))))
        .containsExactly("-Ifoo.proto=out/foo.proto", "--direct_dependencies", "foo.proto");

    assertThat(
            protoArgv(
                ImmutableList.of(
                    derivedArtifact("//:dont-care", "foo.proto"),
                    derivedArtifact("//:dont-care", "bar.proto")) /* directDependencies */,
                ImmutableList.of(derivedArtifact("//:dont-care", "foo.proto"))))
        .containsExactly(
            "-Ifoo.proto=out/foo.proto", "--direct_dependencies", "foo.proto:bar.proto");
  }

  /**
   * Include-maps are the -Ivirtual=physical arguments passed to proto-compiler. When including a
   * file named 'foo/bar.proto' from an external repository 'bla', the include-map should be
   * -Ifoo/bar.proto=external/bla/foo/bar.proto. That is - 'virtual' should be the path relative to
   * the external repo root, and physical should be the physical file location.
   */
  @Test
  public void testIncludeMapsOfExternalFiles() throws Exception {
    assertThat(
            protoArgv(
                null /* protosInDirectoDependencies */,
                ImmutableList.of(artifact("@bla//foo:bar", "external/bla/foo/bar.proto"))))
        .containsExactly("-Ifoo/bar.proto=external/bla/foo/bar.proto");
  }

  // TODO(b/34107586): Fix and enable test.
  @Ignore
  @Test
  public void directDependenciesOnExternalFiles() throws Exception {
    ImmutableList<Artifact> protos =
        ImmutableList.of(artifact("@bla//foo:bar", "external/bla/foo/bar.proto"));
    assertThat(protoArgv(protos, protos)).containsExactly("--direct_dependencies", "foo/bar.proto");
  }

  private Artifact artifact(String ownerLabel, String path) {
    return new Artifact(
        root,
        root.getExecPath().getRelative(path),
        new LabelArtifactOwner(Label.parseAbsoluteUnchecked(ownerLabel)));
  }

  /** Creates a dummy artifact with the given path, that actually resides in /out/<path>. */
  private Artifact derivedArtifact(String ownerLabel, String path) {
    return new Artifact(
        derivedRoot,
        derivedRoot.getExecPath().getRelative(path),
        new LabelArtifactOwner(Label.parseAbsoluteUnchecked(ownerLabel)));
  }

  private static Iterable<String> protoArgv(
      @Nullable Iterable<Artifact> protosInDirectDependencies,
      Iterable<Artifact> transitiveImports) {
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
    NestedSet<Artifact> protosInDirectDependenciesBuilder =
        protosInDirectDependencies != null
            ? NestedSetBuilder.wrap(STABLE_ORDER, protosInDirectDependencies)
            : null;
    NestedSet<Artifact> transitiveImportsNestedSet =
        NestedSetBuilder.wrap(STABLE_ORDER, transitiveImports);
    ProtoCompileActionBuilder.addIncludeMapArguments(
        commandLine,
        protosInDirectDependenciesBuilder,
        NestedSetBuilder.emptySet(STABLE_ORDER),
        transitiveImportsNestedSet);
    return commandLine.build().arguments();
  }
}
