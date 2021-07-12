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
import com.google.devtools.build.skyframe.SkyKey;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ModuleFileFunction}. */
@RunWith(JUnit4.class)
public class ModuleFileFunctionTest extends FoundationTestCase {

  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
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
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, rootDirectory))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
  }

  @Test
  public void testRootModule() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1',compatibility_level=4)",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0',repo_name='see')",
        "single_version_override(module_name='D',version='18')",
        "local_path_override(module_name='E',path='somewhere/else')");
    FakeRegistry registry = registryFactory.newFakeRegistry();
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(ModuleFileValue.keyForRootModule()), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    ModuleFileValue moduleFileValue = result.get(ModuleFileValue.keyForRootModule());
    assertThat(moduleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("A")
                .setVersion("0.1")
                .setCompatibilityLevel(4)
                .addDep("B", ModuleKey.create("B", "1.0"))
                .addDep("see", ModuleKey.create("C", "2.0"))
                .build());
    assertThat(moduleFileValue.getOverrides())
        .containsExactly(
            "D", SingleVersionOverride.create("18", "", ImmutableList.of(), 0),
            "E", LocalPathOverride.create("somewhere/else"));
  }

  @Test
  public void testRootModule_badSelfOverride() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='A')",
        "single_version_override(module_name='A',version='7')");
    FakeRegistry registry = registryFactory.newFakeRegistry();
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(ModuleFileValue.keyForRootModule()), evaluationContext);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString()).contains("invalid override for the root module");
  }

  @Test
  public void testRegistriesCascade() throws Exception {
    // Registry1 has no module B@1.0; registry2 and registry3 both have it. We should be using the
    // B@1.0 from registry2.
    FakeRegistry registry1 = registryFactory.newFakeRegistry();
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B',version='1.0');bazel_dep(name='C',version='2.0')");
    FakeRegistry registry3 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B',version='1.0');bazel_dep(name='D',version='3.0')");
    ModuleFileFunction.REGISTRIES.set(
        differencer, ImmutableList.of(registry1.getUrl(), registry2.getUrl(), registry3.getUrl()));

    SkyKey skyKey = ModuleFileValue.key(ModuleKey.create("B", "1.0"), null);
    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    ModuleFileValue moduleFileValue = result.get(skyKey);
    assertThat(moduleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("C", ModuleKey.create("C", "2.0"))
                .setRegistry(registry2)
                .build());
  }

  // TODO: test local path override

  @Test
  public void testRegistryOverride() throws Exception {
    FakeRegistry registry1 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B',version='1.0',compatibility_level=4)\n"
                    + "bazel_dep(name='C',version='2.0')");
    FakeRegistry registry2 =
        registryFactory
            .newFakeRegistry()
            .addModule(
                ModuleKey.create("B", "1.0"),
                "module(name='B',version='1.0',compatibility_level=6)\n"
                    + "bazel_dep(name='C',version='3.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry1.getUrl()));

    // Override the registry for B to be registry2 (instead of the default registry1).
    SkyKey skyKey =
        ModuleFileValue.key(
            ModuleKey.create("B", "1.0"),
            SingleVersionOverride.create("", registry2.getUrl(), ImmutableList.of(), 0));
    EvaluationResult<ModuleFileValue> result =
        driver.evaluate(ImmutableList.of(skyKey), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    ModuleFileValue moduleFileValue = result.get(skyKey);
    assertThat(moduleFileValue.getModule())
        .isEqualTo(
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .setCompatibilityLevel(6)
                .addDep("C", ModuleKey.create("C", "3.0"))
                .setRegistry(registry2)
                .build());
  }
}
