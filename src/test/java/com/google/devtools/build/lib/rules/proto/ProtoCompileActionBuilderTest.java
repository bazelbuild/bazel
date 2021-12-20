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
import static com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.registerActions;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Exports;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.Services;
import com.google.devtools.build.lib.rules.proto.ProtoCompileActionBuilder.ToolchainInvocation;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OnDemandString;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkValue;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProtoCompileActionBuilder}. */
@RunWith(JUnit4.class)
public class ProtoCompileActionBuilderTest extends BuildViewTestCase {

  private static final InMemoryFileSystem FILE_SYSTEM =
      new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final ArtifactRoot sourceRoot =
      ArtifactRoot.asSourceRoot(Root.fromPath(FILE_SYSTEM.getPath("/")));
  private final ArtifactRoot derivedRoot =
      ArtifactRoot.asDerivedRoot(FILE_SYSTEM.getPath("/"), RootType.Output, "out");

  private AnalysisTestUtil.CollectingAnalysisEnvironment collectingAnalysisEnvironment;
  private Artifact out;

  @Before
  public final void setup() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);

    collectingAnalysisEnvironment =
        new AnalysisTestUtil.CollectingAnalysisEnvironment(getTestAnalysisEnvironment());
    scratch.file(
        "foo/BUILD",
        "package(features = ['-feature'])",
        "proto_library(name = 'bar')",
        "exports_files(['out'])");
    out = getBinArtifactWithNoOwner("out");
  }

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
            "--java_out=param1,param2:%s",
            /* pluginFormatFlag= */ null,
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());
    ProtoLangToolchainProvider toolchainWithPlugin =
        ProtoLangToolchainProvider.create(
            "--PLUGIN_pluginName_out=param3,param4:%s",
            /* pluginFormatFlag= */ "--plugin=protoc-gen-PLUGIN_pluginName=%s",
            plugin,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());
    useConfiguration("--strict_proto_deps=OFF");

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    registerActions(
        ruleContext,
        ImmutableList.of(
            new ToolchainInvocation("dontcare_because_no_plugin", toolchainNoPlugin, "foo.srcjar"),
            new ToolchainInvocation("pluginName", toolchainWithPlugin, "bar.srcjar")),
        protoInfo(
            /* directProtoSources */ ImmutableList.of(protoSource("source_file.proto")),
            /* transitiveProtoSources */ ImmutableList.of(
                protoSource("import1.proto"), protoSource("import2.proto")),
            /* publicImportProtoSources */ ImmutableList.of(),
            /* strictImportableSources */ ImmutableList.of()),
        Label.parseAbsoluteUnchecked("//foo:bar"),
        ImmutableList.of(out),
        "dontcare_because_no_plugin",
        Exports.DO_NOT_USE,
        Services.ALLOW);

    CommandLine cmdLine =
        paramFileCommandLineForAction(
            (SpawnAction) collectingAnalysisEnvironment.getRegisteredActions().get(0));
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
    useConfiguration("--strict_proto_deps=OFF");

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    registerActions(
        ruleContext,
        /* toolchainInvocations= */ ImmutableList.of(),
        protoInfo(
            /* directProtoSources */ ImmutableList.of(
                protoSource(derivedArtifact("source_file.proto"))),
            /* transitiveProtoSources */ ImmutableList.of(),
            /* publicImportProtoSources */ ImmutableList.of(),
            /* strictImportableSources */ ImmutableList.of()),
        Label.parseAbsoluteUnchecked("//foo:bar"),
        ImmutableList.of(out),
        "dontcare_because_no_plugin",
        Exports.DO_NOT_USE,
        Services.ALLOW);

    CommandLine cmdLine =
        paramFileCommandLineForAction(
            (SpawnAction) collectingAnalysisEnvironment.getRegisteredActions().get(0));
    assertThat(cmdLine.arguments()).containsExactly("out/source_file.proto");
  }

  @Test
  public void commandLine_strictDeps() throws Exception {
    ProtoLangToolchainProvider toolchain =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:%s",
            /* pluginFormatFlag= */ null,
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    registerActions(
        ruleContext,
        ImmutableList.of(new ToolchainInvocation("dontcare", toolchain, "foo.srcjar")),
        protoInfo(
            /* directProtoSources */ ImmutableList.of(protoSource("source_file.proto")),
            /* transitiveProtoSources */ ImmutableList.of(
                protoSource("import1.proto"), protoSource("import2.proto")),
            /* publicImportProtoSources */ ImmutableList.of(),
            /* strictImportableSources */ ImmutableList.of(protoSource("import1.proto"))),
        Label.parseAbsoluteUnchecked("//foo:bar"),
        ImmutableList.of(out),
        "dontcare_because_no_plugin",
        Exports.DO_NOT_USE,
        Services.ALLOW);

    CommandLine cmdLine =
        paramFileCommandLineForAction(
            (SpawnAction) collectingAnalysisEnvironment.getRegisteredActions().get(0));
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
            "--java_out=param1,param2:%s",
            /* pluginFormatFlag= */ null,
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());
    useConfiguration(
        "--strict_proto_deps=OFF", "--experimental_java_proto_add_allowed_public_imports");

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    registerActions(
        ruleContext,
        ImmutableList.of(new ToolchainInvocation("dontcare", toolchain, "foo.srcjar")),
        protoInfo(
            /* directProtoSources */ ImmutableList.of(protoSource("source_file.proto")),
            /* transitiveProtoSources */ ImmutableList.of(
                protoSource("import1.proto"), protoSource("import2.proto")),
            /* publicImportProtoSources */ ImmutableList.of(protoSource("export1.proto")),
            /* strictImportableSources */ ImmutableList.of()),
        Label.parseAbsoluteUnchecked("//foo:bar"),
        ImmutableList.of(out),
        "dontcare_because_no_plugin",
        Exports.USE,
        Services.ALLOW);

    CommandLine cmdLine =
        paramFileCommandLineForAction(
            (SpawnAction) collectingAnalysisEnvironment.getRegisteredActions().get(0));
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
    useConfiguration("--protocopt=--foo", "--protocopt=--bar");

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    registerActions(
        ruleContext,
        ImmutableList.of(),
        protoInfo(
            /* directProtoSources */ ImmutableList.of(),
            /* transitiveProtoSources */ ImmutableList.of(),
            /* publicImportProtoSources */ ImmutableList.of(),
            /* strictImportableSources */ ImmutableList.of()),
        Label.parseAbsoluteUnchecked("//foo:bar"),
        ImmutableList.of(out),
        "flavour",
        Exports.DO_NOT_USE,
        Services.DISALLOW);

    CommandLine cmdLine =
        paramFileCommandLineForAction(
            (SpawnAction) collectingAnalysisEnvironment.getRegisteredActions().get(0));
    assertThat(cmdLine.arguments()).containsAtLeast("--disallow_services", "--foo", "--bar");
  }

  private static class InterceptOnDemandString extends OnDemandString implements StarlarkValue {
    boolean hasBeenCalled;

          @Override
          public String toString() {
      hasBeenCalled = true;
            return "mu";
          }
  }

  @Test
  public void outReplacementAreLazilyEvaluated() throws Exception {
    InterceptOnDemandString outReplacement = new InterceptOnDemandString();
    ProtoLangToolchainProvider toolchain =
        ProtoLangToolchainProvider.create(
            "--java_out=param1,param2:%s",
            /* pluginFormatFlag= */ null,
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    registerActions(
        ruleContext,
        ImmutableList.of(new ToolchainInvocation("pluginName", toolchain, outReplacement)),
        protoInfo(
            /* directProtoSources */ ImmutableList.of(),
            /* transitiveProtoSources */ ImmutableList.of(),
            /* publicImportProtoSources */ ImmutableList.of(),
            /* strictImportableSources */ ImmutableList.of()),
        Label.parseAbsoluteUnchecked("//foo:bar"),
        ImmutableList.of(out),
        "flavour",
        Exports.DO_NOT_USE,
        Services.ALLOW);

    CommandLine cmdLine =
        paramFileCommandLineForAction(
            (SpawnAction) collectingAnalysisEnvironment.getRegisteredActions().get(0));
    assertThat(outReplacement.hasBeenCalled).isFalse();

    cmdLine.arguments();
    assertThat(outReplacement.hasBeenCalled).isTrue();
  }

  /**
   * Tests that if the same invocation-name is specified by more than one invocation,
   * ProtoCompileActionBuilder throws an exception.
   */
  @Test
  public void exceptionIfSameName() throws Exception {
    ProtoLangToolchainProvider toolchain1 =
        ProtoLangToolchainProvider.create(
            "dontcare=%s",
            /* pluginFormatFlag= */ null,
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());
    ProtoLangToolchainProvider toolchain2 =
        ProtoLangToolchainProvider.create(
            "dontcare=%s",
            /* pluginFormatFlag= */ null,
            /* pluginExecutable= */ null,
            /* runtime= */ mock(TransitiveInfoCollection.class),
            /* providedProtoSources= */ ImmutableList.of());

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget("//foo:bar"), collectingAnalysisEnvironment);
    IllegalStateException e =
        assertThrows(
            IllegalStateException.class,
            () ->
                registerActions(
                    ruleContext,
                    ImmutableList.of(
                        new ToolchainInvocation("pluginName", toolchain1, "outReplacement"),
                        new ToolchainInvocation("pluginName", toolchain2, "outReplacement")),
                    protoInfo(
                        /* directProtoSources */ ImmutableList.of(),
                        /* transitiveProtoSources */ ImmutableList.of(),
                        /* publicImportProtoSources */ ImmutableList.of(),
                        /* strictImportableSources */ ImmutableList.of()),
                    Label.parseAbsoluteUnchecked("//foo:bar"),
                    ImmutableList.of(out),
                    "flavour",
                    Exports.DO_NOT_USE,
                    Services.ALLOW));

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
        sourceRoot,
        sourceRoot.getExecPath().getRelative(path),
        new LabelArtifactOwner(Label.parseAbsoluteUnchecked(ownerLabel)));
  }

  /** Creates a dummy artifact with the given path, that actually resides in /out/<path>. */
  private Artifact derivedArtifact(String path) {
    Artifact.DerivedArtifact derivedArtifact =
        DerivedArtifact.create(
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

  @Test
  public void testEstimateResourceConsumptionLocal() throws Exception {

    assertThat(
            new ProtoCompileActionBuilder.ProtoCompileResourceSetBuilder()
                .buildResourceSet(OS.DARWIN, 0))
        .isEqualTo(ResourceSet.createWithRamCpu(25, 1));

    assertThat(
            new ProtoCompileActionBuilder.ProtoCompileResourceSetBuilder()
                .buildResourceSet(OS.LINUX, 2))
        .isEqualTo(ResourceSet.createWithRamCpu(25.3, 1));
  }
}
