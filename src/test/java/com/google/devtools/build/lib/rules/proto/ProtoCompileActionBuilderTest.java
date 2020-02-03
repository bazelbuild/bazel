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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Deps;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import javax.annotation.Nullable;
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

  private ProtoInfo protoInfo(
      ImmutableList<Artifact> directProtos,
      NestedSet<Artifact> transitiveProtos,
      NestedSet<String> transitiveProtoSourceRoots,
      NestedSet<String> strictImportableProtoSourceRoots,
      NestedSet<Pair<Artifact, String>> strictImportableProtos,
      NestedSet<Pair<Artifact, String>> exportedProtos) {
    return new ProtoInfo(
        directProtos,
        directProtos,
        /* directProtoSourceRoot= */ "",
        transitiveProtos,
        transitiveProtos,
        transitiveProtoSourceRoots,
        /* strictImportableProtosForDependents */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        strictImportableProtos,
        strictImportableProtos,
        strictImportableProtoSourceRoots,
        exportedProtos,
        /* exportedProtoSourceRoots */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        artifact("//:direct-descriptor-set", "direct-descriptor-set"),
        /* getTransitiveDescriptorSets */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        Location.BUILTIN);
  }

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

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(
                new ToolchainInvocation(
                    "dontcare_because_no_plugin", toolchainNoPlugin, "foo.srcjar"),
                new ToolchainInvocation("pluginName", toolchainWithPlugin, "bar.srcjar")),
            "bazel-out",
            protoInfo(
                /* directProtos */ ImmutableList.of(artifact("//:dont-care", "source_file.proto")),
                /* transitiveProtos */ NestedSetBuilder.create(
                    STABLE_ORDER,
                    artifact("//:dont-care", "import1.proto"),
                    artifact("//:dont-care", "import2.proto")),

                /* transitiveProtoSourceRoots= */ NestedSetBuilder.create(STABLE_ORDER, "."),
                /* strictImportableProtoSourceRoots= */ NestedSetBuilder.create(
                    Order.STABLE_ORDER, "."),
                /* strictImportableProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* exportedProtos = */ NestedSetBuilder.emptySet(STABLE_ORDER)),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.NON_STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
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
    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            /* toolchainInvocations= */ ImmutableList.of(),
            "bazel-out",
            protoInfo(
                /* directProtos */ ImmutableList.of(derivedArtifact("source_file.proto")),
                /* transitiveProtos */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* transitiveProtoSourceRoots= */ NestedSetBuilder.create(Order.STABLE_ORDER, "."),
                /* strictImportableProtoSourceRoots= */ NestedSetBuilder.create(
                    Order.STABLE_ORDER, "."),
                /* strictImportableProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* exportedProtos = */ NestedSetBuilder.emptySet(STABLE_ORDER)),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.NON_STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
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

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(new ToolchainInvocation("dontcare", toolchain, "foo.srcjar")),
            "bazel-out",
            protoInfo(
                /* directProtos */ ImmutableList.of(artifact("//:dont-care", "source_file.proto")),
                /* transitiveProtos */ NestedSetBuilder.create(
                    STABLE_ORDER,
                    artifact("//:dont-care", "import1.proto"),
                    artifact("//:dont-care", "import2.proto")),

                /* transitiveProtoSourceRoots= */ NestedSetBuilder.create(STABLE_ORDER, "."),
                /* strictImportableProtoSourceRoots= */ NestedSetBuilder.create(STABLE_ORDER, "."),
                NestedSetBuilder.create(
                    STABLE_ORDER,
                    Pair.of(artifact("//:dont-care", "import1.proto"), "import1.proto")),
                /* exportedProtos = */ NestedSetBuilder.emptySet(STABLE_ORDER)),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
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
  public void commandLine_exports() throws Exception {
    ProtoLangToolchainProvider toolchain =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:$(OUT)",
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* blacklistedProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER));

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(new ToolchainInvocation("dontcare", toolchain, "foo.srcjar")),
            "bazel-out",
            protoInfo(
                /* directProtos */ ImmutableList.of(artifact("//:dont-care", "source_file.proto")),
                /* transitiveProtos */ NestedSetBuilder.create(
                    STABLE_ORDER,
                    artifact("//:dont-care", "import1.proto"),
                    artifact("//:dont-care", "import2.proto")),

                /* transitiveProtoSourceRoots= */ NestedSetBuilder.create(STABLE_ORDER, "."),
                /* strictImportableProtoSourceRoots= */ NestedSetBuilder.create(STABLE_ORDER, "."),
                /* strictImportableProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* exportedProtos = */ NestedSetBuilder.create(
                    STABLE_ORDER,
                    Pair.of(artifact("//:dont-care", "foo/export1.proto"), "export1.proto"))),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.NON_STRICT,
            Exports.USE,
            Services.ALLOW,
            /* protocOpts= */ ImmutableList.of());

    assertThat(cmdLine.arguments())
        .containsExactly(
            "--java_out=param1,param2:foo.srcjar",
            "-Iimport1.proto=import1.proto",
            "-Iimport2.proto=import2.proto",
            "--allowed_public_imports",
            "export1.proto",
            "source_file.proto")
        .inOrder();
  }

  @Test
  public void otherParameters() throws Exception {
    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(),
            "bazel-out",
            protoInfo(
                /* directProtos */ ImmutableList.of(),
                /* transitiveProtos */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* transitiveProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* strictImportableProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* strictImportableProtos */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* exportedProtos = */ NestedSetBuilder.emptySet(STABLE_ORDER)),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.STRICT,
            Exports.DO_NOT_USE,
            Services.DISALLOW,
            /* protocOpts= */ ImmutableList.of("--foo", "--bar"));

    assertThat(cmdLine.arguments()).containsAtLeast("--disallow_services", "--foo", "--bar");
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

    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            ImmutableList.of(new ToolchainInvocation("pluginName", toolchain, outReplacement)),
            "bazel-out",
            protoInfo(
                /* directProtos*/ ImmutableList.of(),
                /* transitiveProtos */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* transitiveProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* strictImportableProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* strictImportableProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                /* exportedProtos = */ NestedSetBuilder.emptySet(STABLE_ORDER)),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
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

    IllegalStateException e =
        assertThrows(
            IllegalStateException.class,
            () ->
                createCommandLineFromToolchains(
                    ImmutableList.of(
                        new ToolchainInvocation("pluginName", toolchain1, "outReplacement"),
                        new ToolchainInvocation("pluginName", toolchain2, "outReplacement")),
                    "bazel-out",
                    protoInfo(
                        /* directProtos */ ImmutableList.of(),
                        /* transitiveProtos */ NestedSetBuilder.emptySet(STABLE_ORDER),
                        /* transitiveProtoSourceRoots= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                        /* strictImportableProtoSourceRoots= */ NestedSetBuilder.emptySet(
                            STABLE_ORDER),
                        /* strictImportableProtos= */ NestedSetBuilder.emptySet(STABLE_ORDER),
                        /* exportedProtos = */ NestedSetBuilder.emptySet(STABLE_ORDER)),
                    Label.parseAbsoluteUnchecked("//foo:bar"),
                    Deps.STRICT,
                    Exports.DO_NOT_USE,
                    Services.ALLOW,
                    /* protocOpts= */ ImmutableList.of()));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Invocation name pluginName appears more than once. "
                + "This could lead to incorrect proto-compiler behavior");
  }

  @Test
  public void testProtoCommandLineArgv() throws Exception {
    assertThat(
            protoArgv(
                null /* directDependencies */,
                ImmutableList.of(derivedArtifact("foo.proto")),
                ImmutableList.of(".")))
        .containsExactly("-Ifoo.proto=out/foo.proto");

    assertThat(
            protoArgv(
                ImmutableList.of() /* directDependencies */,
                ImmutableList.of(derivedArtifact("foo.proto")),
                ImmutableList.of(".")))
        .containsExactly("-Ifoo.proto=out/foo.proto", "--direct_dependencies=");

    assertThat(
            protoArgv(
                ImmutableList.of(
                    Pair.of(derivedArtifact("foo.proto"), null)) /* directDependencies */,
                ImmutableList.of(derivedArtifact("foo.proto")),
                ImmutableList.of(".")))
        .containsExactly("-Ifoo.proto=out/foo.proto", "--direct_dependencies", "foo.proto");

    assertThat(
            protoArgv(
                ImmutableList.of(
                    Pair.of(derivedArtifact("foo.proto"), null),
                    Pair.of(derivedArtifact("bar.proto"), null)) /* directDependencies */,
                ImmutableList.of(derivedArtifact("foo.proto")),
                ImmutableList.of(".")))
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
                ImmutableList.of(artifact("@bla//foo:bar", "external/bla/foo/bar.proto")),
                ImmutableList.of("external/bla")))
        .containsExactly("-Ifoo/bar.proto=external/bla/foo/bar.proto");
  }

  @Test
  public void directDependenciesOnExternalFiles() throws Exception {
    ImmutableList<Artifact> protos =
        ImmutableList.of(artifact("@bla//foo:bar", "external/bla/foo/bar.proto"));
    ImmutableList<Pair<Artifact, String>> protosImports =
        ImmutableList.of(Pair.of(artifact("@bla//foo:bar", "external/bla/foo/bar.proto"), null));
    assertThat(protoArgv(protosImports, protos, ImmutableList.of("external/bla")))
        .containsExactly(
            "-Ifoo/bar.proto=external/bla/foo/bar.proto", "--direct_dependencies", "foo/bar.proto");
  }

  private Artifact artifact(String ownerLabel, String path) {
    return new Artifact.SourceArtifact(
        root,
        root.getExecPath().getRelative(path),
        new LabelArtifactOwner(Label.parseAbsoluteUnchecked(ownerLabel)));
  }

  /** Creates a dummy artifact with the given path, that actually resides in /out/<path>. */
  private Artifact derivedArtifact(String path) {
    Artifact.DerivedArtifact derivedArtifact =
        new Artifact.DerivedArtifact(
            derivedRoot,
            derivedRoot.getExecPath().getRelative(path),
            ActionsTestUtil.NULL_ARTIFACT_OWNER);
    derivedArtifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    return derivedArtifact;
  }

  private static Iterable<String> protoArgv(
      @Nullable Iterable<Pair<Artifact, String>> protosInDirectDependencies,
      Iterable<Artifact> transitiveImports,
      Iterable<String> protoSourceRoots) {
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
    NestedSet<Pair<Artifact, String>> protosInDirectDependenciesBuilder =
        protosInDirectDependencies != null
            ? NestedSetBuilder.wrap(STABLE_ORDER, protosInDirectDependencies)
            : null;
    NestedSet<Artifact> transitiveImportsNestedSet =
        NestedSetBuilder.wrap(STABLE_ORDER, transitiveImports);
    ProtoCompileActionBuilder.addIncludeMapArguments(
        "blaze-out",
        commandLine,
        protosInDirectDependenciesBuilder,
        NestedSetBuilder.wrap(Order.STABLE_ORDER, protoSourceRoots),
        transitiveImportsNestedSet);
    return commandLine.build().arguments();
  }
}
