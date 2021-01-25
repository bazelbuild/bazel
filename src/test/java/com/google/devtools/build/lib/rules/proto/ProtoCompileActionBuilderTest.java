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
import static org.junit.Assert.assertThrows;
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
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Deps;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProtoCompileActionBuilder}. */
@RunWith(JUnit4.class)
public class ProtoCompileActionBuilderTest {

  private static final InMemoryFileSystem FILE_SYSTEM =
      new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final ArtifactRoot root =
      ArtifactRoot.asSourceRoot(Root.fromPath(FILE_SYSTEM.getPath("/")));
  private final ArtifactRoot derivedRoot =
      ArtifactRoot.asDerivedRoot(FILE_SYSTEM.getPath("/"), false, false, false, "out");

  private ProtoSource protoSource(String importPath) {
    return protoSource(artifact("//:dont-care", importPath));
  }

  private ProtoSource protoSource(Artifact protoSource) {
    return protoSource(protoSource, PathFragment.EMPTY_FRAGMENT);
  }

  private ProtoSource protoSource(Artifact protoSource, PathFragment sourceRoot) {
    return new ProtoSource(protoSource, sourceRoot);
  }

  private ProtoInfo protoInfo(
      ImmutableList<ProtoSource> directProtoSources,
      ImmutableList<ProtoSource> transitiveProtoSources,
      ImmutableList<ProtoSource> publicImportProtoSources,
      ImmutableList<ProtoSource> strictImportableSources) {
    return new ProtoInfo(
        /* directSources */ directProtoSources,
        /* directProtoSourceRoot */ PathFragment.EMPTY_FRAGMENT,
        /* transitiveSources */ NestedSetBuilder.wrap(Order.STABLE_ORDER, transitiveProtoSources),
        /* transitiveProtoSources */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /* originalTransitiveProtoSources */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /* transitiveProtoSourceRoots */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /* strictImportableProtoSourcesForDependents */ NestedSetBuilder.emptySet(
            Order.STABLE_ORDER),
        /* directDescriptorSet */ artifact("//:direct-descriptor-set", "direct-descriptor-set"),
        /* transitiveDescriptorSets */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /* exportedSources */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /* strictImportableSources */ NestedSetBuilder.wrap(
            Order.STABLE_ORDER, strictImportableSources),
        /* publicImportSources */ NestedSetBuilder.wrap(
            Order.STABLE_ORDER, publicImportProtoSources));
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
                /* directProtoSources */ ImmutableList.of(protoSource("source_file.proto")),
                /* transitiveProtoSources */ ImmutableList.of(
                    protoSource("import1.proto"), protoSource("import2.proto")),
                /* publicImportProtoSources */ ImmutableList.of(),
                /* strictImportableSources */ ImmutableList.of()),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.NON_STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
            /* protocOpts= */ ImmutableList.of(),
            false);

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
  public void commandline_derivedArtifact() throws Exception {
    // Verify that the command line contains the correct path to a generated protocol buffers.
    CustomCommandLine cmdLine =
        createCommandLineFromToolchains(
            /* toolchainInvocations= */ ImmutableList.of(),
            "bazel-out",
            protoInfo(
                /* directProtoSources */ ImmutableList.of(
                    protoSource(derivedArtifact("source_file.proto"))),
                /* transitiveProtoSources */ ImmutableList.of(),
                /* publicImportProtoSources */ ImmutableList.of(),
                /* strictImportableSources */ ImmutableList.of()),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.NON_STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
            /* protocOpts= */ ImmutableList.of(),
            false);

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
                /* directProtoSources */ ImmutableList.of(protoSource("source_file.proto")),
                /* transitiveProtoSources */ ImmutableList.of(
                    protoSource("import1.proto"), protoSource("import2.proto")),
                /* publicImportProtoSources */ ImmutableList.of(),
                /* strictImportableSources */ ImmutableList.of(protoSource("import1.proto"))),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
            /* protocOpts= */ ImmutableList.of(),
            false);

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
                /* directProtoSources */ ImmutableList.of(protoSource("source_file.proto")),
                /* transitiveProtoSources */ ImmutableList.of(
                    protoSource("import1.proto"), protoSource("import2.proto")),
                /* publicImportProtoSources */ ImmutableList.of(protoSource("export1.proto")),
                /* strictImportableSources */ ImmutableList.of()),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.NON_STRICT,
            Exports.USE,
            Services.ALLOW,
            /* protocOpts= */ ImmutableList.of(),
            false);

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
                /* directProtoSources */ ImmutableList.of(),
                /* transitiveProtoSources */ ImmutableList.of(),
                /* publicImportProtoSources */ ImmutableList.of(),
                /* strictImportableSources */ ImmutableList.of()),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.STRICT,
            Exports.DO_NOT_USE,
            Services.DISALLOW,
            /* protocOpts= */ ImmutableList.of("--foo", "--bar"),
            false);

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
                /* directProtoSources */ ImmutableList.of(),
                /* transitiveProtoSources */ ImmutableList.of(),
                /* publicImportProtoSources */ ImmutableList.of(),
                /* strictImportableSources */ ImmutableList.of()),
            Label.parseAbsoluteUnchecked("//foo:bar"),
            Deps.STRICT,
            Exports.DO_NOT_USE,
            Services.ALLOW,
            /* protocOpts= */ ImmutableList.of(),
            false);

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
                        /* directProtoSources */ ImmutableList.of(),
                        /* transitiveProtoSources */ ImmutableList.of(),
                        /* publicImportProtoSources */ ImmutableList.of(),
                        /* strictImportableSources */ ImmutableList.of()),
                    Label.parseAbsoluteUnchecked("//foo:bar"),
                    Deps.STRICT,
                    Exports.DO_NOT_USE,
                    Services.ALLOW,
                    /* protocOpts= */ ImmutableList.of(),
                    false));
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
                /* transitiveSources */ ImmutableList.of(
                    protoSource(derivedArtifact("foo.proto"), derivedRoot.getExecPath())),
                /* importableProtoSources */ null))
        .containsExactly("-Ifoo.proto=out/foo.proto");

    assertThat(
            protoArgv(
                /* transitiveSources */ ImmutableList.of(
                    protoSource(derivedArtifact("foo.proto"), derivedRoot.getExecPath())),
                /* importableProtoSources */ ImmutableList.of()))
        .containsExactly("-Ifoo.proto=out/foo.proto", "--direct_dependencies=");

    assertThat(
            protoArgv(
                /* transitiveSources */ ImmutableList.of(
                    protoSource(derivedArtifact("foo.proto"), derivedRoot.getExecPath())),
                /* importableProtoSources */ ImmutableList.of(
                    protoSource(derivedArtifact("foo.proto"), derivedRoot.getExecPath()))))
        .containsExactly("-Ifoo.proto=out/foo.proto", "--direct_dependencies", "foo.proto");

    assertThat(
            protoArgv(
                /* transitiveSources */ ImmutableList.of(
                    protoSource(derivedArtifact("foo.proto"), derivedRoot.getExecPath())),
                /* importableProtoSources */ ImmutableList.of(
                    protoSource(derivedArtifact("foo.proto"), derivedRoot.getExecPath()),
                    protoSource(derivedArtifact("bar.proto"), derivedRoot.getExecPath()))))
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
                /* transitiveSources */ ImmutableList.of(
                    protoSource(
                        artifact("@bla//foo:bar", "external/bla/foo/bar.proto"),
                        PathFragment.create("external/bla"))),
                /* importableProtoSources */ ImmutableList.of()))
        .containsExactly("-Ifoo/bar.proto=external/bla/foo/bar.proto", "--direct_dependencies=");
  }

  @Test
  public void directDependenciesOnExternalFiles() throws Exception {
    Artifact protoSource = artifact("@bla//foo:bar", "external/bla/foo/bar.proto");
    assertThat(
            protoArgv(
                /* transitiveSources */ ImmutableList.of(
                    protoSource(protoSource, PathFragment.create("external/bla"))),
                /* importableProtoSources */ ImmutableList.of(
                    protoSource(protoSource, PathFragment.create("external/bla")))))
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
      Iterable<ProtoSource> transitiveSources,
      @Nullable Iterable<ProtoSource> importableProtoSources)
      throws Exception {
    CustomCommandLine.Builder commandLine = CustomCommandLine.builder();
    NestedSet<ProtoSource> importableProtoSourceSet =
        importableProtoSources != null
            ? NestedSetBuilder.wrap(STABLE_ORDER, importableProtoSources)
            : null;
    ProtoCompileActionBuilder.addIncludeMapArguments(
        commandLine,
        importableProtoSourceSet,
        NestedSetBuilder.wrap(STABLE_ORDER, transitiveSources));
    return commandLine.build().arguments();
  }
}
