// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static org.junit.Assert.fail;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileStateKey;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelLockFileFunction}. */
@RunWith(JUnit4.class)
public class BazelLockFileFunctionTest extends FoundationTestCase {

  private MemoizingEvaluator evaluator;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
    differencer = new SequencedRecordingDifferencer();
    registryFactory = new FakeRegistry.Factory();
    evaluationContext =
        EvaluationContext.newBuilder().setParallelism(8).setEventHandler(reporter).build();

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

    evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(FileValue.FILE, new FileFunction(packageLocator, directories))
                .put(
                    FileStateKey.FILE_STATE,
                    new FileStateFunction(
                        Suppliers.ofInstance(
                            new TimestampGranularityMonitor(BlazeClock.instance())),
                        SyscallCache.NO_CACHE,
                        externalFilesHelper))
                .put(SkyFunctions.BAZEL_LOCK_FILE, new BazelLockFileFunction(rootDirectory, FakeRegistry.class))
                .buildOrThrow(),
            differencer);

    PrecomputedValue.STARLARK_SEMANTICS.set(
        differencer,
        StarlarkSemantics.builder().setBool(BuildLanguageOptions.ENABLE_BZLMOD, true).build());
  }

  @Test
  public void simpleModule() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.ROOT,
                Module.builder()
                    .setName("my_root")
                    .setVersion(Version.parse("1.0"))
                    .setKey(ModuleKey.ROOT)
                    .build())
            .buildOrThrow();

    BazelLockFileFunction.updateLockedModule("moduleHash", depGraph);
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    BazelLockFileValue value = result.get(BazelLockFileValue.KEY);
    Assert.assertEquals(value.getModuleDepGraph(), depGraph);
  }

  @Test
  public void fullModule() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='my_root', version='1.0')");

    //Populating different types of the module properties to make sure the serialize/deserialize
    //is working properly with the auto_value_gson and that we have the needed adapters
    Map<String, String> imports =
        Map.ofEntries(Map.entry("Key1","1"), Map.entry("Key2","2"));
    Map<String, ModuleKey> excPlatforms =
        Map.ofEntries(Map.entry("MapKey", createModuleKey("root", "1.0")));
    Dict<String, Object> attribs =
        Dict.<String, Object>builder().put("DictKey", "Value").buildImmutable();

    Tag myTag = Tag.builder()
        .setTagName("tagName")
        .setAttributeValues(attribs)
        .setLocation(new Location("Location", 1,2))
        .build();

    List<ModuleExtensionUsage> extensionUsages =
        Arrays.asList(ModuleExtensionUsage.builder()
                .setExtensionName("name")
                .setExtensionBzlFile("fileName")
                .setTags(ImmutableList.of(myTag))
                .setImports(ImmutableBiMap.copyOf(imports))
                .setLocation(new Location("String", 1,2))
            .build());

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("reg_dep_1", "1.0"),
                "module(name='reg_dep_1', version='1.0');");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.ROOT,
                Module.builder()
                    .setName("my_root")
                    .setVersion(Version.parse("1.0"))
                    .setKey(ModuleKey.ROOT)
                    .setRegistry(registry)
                    .setOriginalDeps(ImmutableMap.copyOf(excPlatforms))
                    .setExtensionUsages(ImmutableList.copyOf(extensionUsages))
                    .addDep("reg_dep_1", createModuleKey("dep_1", ""))
                    .build())
            .buildOrThrow();

    BazelLockFileFunction.updateLockedModule("moduleHash", depGraph);

    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    BazelLockFileValue value = result.get(BazelLockFileValue.KEY);
    Assert.assertEquals(value.getModuleDepGraph(), depGraph);
  }

  @Test
  public void moduleWithNestedDeps() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0')");

    // Root depends on dep@1.0 and dep@2.0 at the same time with a multiple-version override.
    // Root also depends on rules_cc as a normal dep.
    // dep@1.0 depends on rules_java, which is overridden by a non-registry override (see below).
    ImmutableMap<ModuleKey, Module> depGraph =
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.ROOT,
                Module.builder()
                    .setName("my_root")
                    .setVersion(Version.parse("1.0"))
                    .setKey(ModuleKey.ROOT)
                    .addDep("my_dep_1", createModuleKey("dep", "1.0"))
                    .addDep("my_dep_2", createModuleKey("dep", "2.0"))
                    .addDep("rules_cc", createModuleKey("rules_cc", "1.0"))
                    .build())
            .put(
                createModuleKey("dep", "1.0"),
                Module.builder()
                    .setName("dep")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("dep", "1.0"))
                    .addDep("rules_java", createModuleKey("rules_java", ""))
                    .build())
            .put(
                createModuleKey("dep", "2.0"),
                Module.builder()
                    .setName("dep")
                    .setVersion(Version.parse("2.0"))
                    .setKey(createModuleKey("dep", "2.0"))
                    .build())
            .put(
                createModuleKey("rules_cc", "1.0"),
                Module.builder()
                    .setName("rules_cc")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("rules_cc", "1.0"))
                    .build())
            .put(
                createModuleKey("rules_java", ""),
                Module.builder()
                    .setName("rules_java")
                    .setVersion(Version.parse("1.0"))
                    .setKey(createModuleKey("rules_java", ""))
                    .build())
            .buildOrThrow();

    BazelLockFileFunction.updateLockedModule("moduleHash", depGraph);
    EvaluationResult<BazelLockFileValue> result =
        evaluator.evaluate(ImmutableList.of(BazelLockFileValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    BazelLockFileValue value = result.get(BazelLockFileValue.KEY);
    Assert.assertEquals(value.getModuleDepGraph(), depGraph);
  }

}
