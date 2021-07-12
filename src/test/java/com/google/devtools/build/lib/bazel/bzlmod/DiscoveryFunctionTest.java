// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DiscoveryFunction}. */
@RunWith(JUnit4.class)
public class DiscoveryFunctionTest extends FoundationTestCase {

  private Path workspaceRoot;
  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
    workspaceRoot = scratch.dir("/ws");
    differencer = new SequencedRecordingDifferencer();
    evaluationContext =
        EvaluationContext.newBuilder().setNumThreads(8).setEventHandler(reporter).build();
    registryFactory = new FakeRegistry.Factory();
    AtomicReference<PathPackageLocator> packageLocator =
        new AtomicReference<>(
            new PathPackageLocator(
                outputBase,
                ImmutableList.of(Root.fromPath(rootDirectory)),
                BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY));
    BlazeDirectories directories =
        new BlazeDirectories(
            new ServerDirectories(rootDirectory, outputBase, rootDirectory),
            rootDirectory,
            /* defaultSystemJavabase= */ null,
            AnalysisMock.get().getProductName());
    ExternalFilesHelper externalFilesHelper =
        ExternalFilesHelper.createForTesting(
            packageLocator,
            ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            directories);

    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(FileValue.FILE, new FileFunction(packageLocator))
                .put(
                    FileStateValue.FILE_STATE,
                    new FileStateFunction(
                        new AtomicReference<TimestampGranularityMonitor>(),
                        new AtomicReference<>(UnixGlob.DEFAULT_SYSCALLS),
                        externalFilesHelper))
                .put(SkyFunctions.DISCOVERY, new DiscoveryFunction())
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, workspaceRoot))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
  }

  @Test
  public void testSimpleDiamond() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='D',version='3.0')")
            .addModule(
                ModuleKey.create("C", "2.0"),
                "module(name='C', version='2.0');bazel_dep(name='D',version='3.0')")
            .addModule(ModuleKey.create("D", "3.0"), "module(name='D', version='3.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("0.1")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .addDep("C", ModuleKey.create("C", "2.0"))
                .build(),
            ModuleKey.create("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("D", ModuleKey.create("D", "3.0"))
                .setRegistry(registry)
                .build(),
            ModuleKey.create("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion("2.0")
                .addDep("D", ModuleKey.create("D", "3.0"))
                .setRegistry(registry)
                .build(),
            ModuleKey.create("D", "3.0"),
            Module.builder().setName("D").setVersion("3.0").setRegistry(registry).build());
  }

  @Test
  public void testCircularDependency() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='C',version='2.0')")
            .addModule(
                ModuleKey.create("C", "2.0"),
                "module(name='C', version='2.0');bazel_dep(name='B',version='1.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("0.1")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .build(),
            ModuleKey.create("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("C", ModuleKey.create("C", "2.0"))
                .setRegistry(registry)
                .build(),
            ModuleKey.create("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion("2.0")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .setRegistry(registry)
                .build());
  }

  @Test
  public void testCircularDependencyOnRootModule() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='A',version='2.0')")
            .addModule(ModuleKey.create("A", "2.0"), "module(name='A', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("0.1")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .build(),
            ModuleKey.create("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("A", ModuleKey.create("A", ""))
                .setRegistry(registry)
                .build());
  }

  @Test
  public void testSingleVersionOverride() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='0.1')",
        "single_version_override(module_name='C',version='2.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "0.1"),
                "module(name='B', version='0.1');bazel_dep(name='C',version='1.0')")
            .addModule(ModuleKey.create("C", "1.0"), "module(name='C', version='1.0');")
            .addModule(ModuleKey.create("C", "2.0"), "module(name='C', version='2.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("0.1")
                .addDep("B", ModuleKey.create("B", "0.1"))
                .build(),
            ModuleKey.create("B", "0.1"),
            Module.builder()
                .setName("B")
                .setVersion("0.1")
                .addDep("C", ModuleKey.create("C", "2.0"))
                .setRegistry(registry)
                .build(),
            ModuleKey.create("C", "2.0"),
            Module.builder().setName("C").setVersion("2.0").setRegistry(registry).build());
  }

  @Test
  public void testRegistryOverride() throws Exception {
    FakeRegistry registry1 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "0.1"),
                "module(name='B', version='0.1');bazel_dep(name='C',version='1.0')")
            .addModule(ModuleKey.create("C", "1.0"), "module(name='C', version='1.0');");
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("C", "1.0"),
                "module(name='C', version='1.0');bazel_dep(name='B',version='0.1')");
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='0.1')",
        "single_version_override(module_name='C',registry='" + registry2.getUrl() + "')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry1.getUrl()));

    EvaluationResult<DiscoveryValue> result =
        driver.evaluate(ImmutableList.of(DiscoveryValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    DiscoveryValue discoveryValue = result.get(DiscoveryValue.KEY);
    assertThat(discoveryValue.getRootModuleName()).isEqualTo("A");
    assertThat(discoveryValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("0.1")
                .addDep("B", ModuleKey.create("B", "0.1"))
                .build(),
            ModuleKey.create("B", "0.1"),
            Module.builder()
                .setName("B")
                .setVersion("0.1")
                .addDep("C", ModuleKey.create("C", "1.0"))
                .setRegistry(registry1)
                .build(),
            ModuleKey.create("C", "1.0"),
            Module.builder()
                .setName("C")
                .setVersion("1.0")
                .addDep("B", ModuleKey.create("B", "0.1"))
                .setRegistry(registry2)
                .build());
  }

  // TODO(wyv): test local path override
}
