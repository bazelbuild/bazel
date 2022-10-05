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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.BazelSkyframeExecutorConstants;
import com.google.devtools.build.lib.skyframe.ClientEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.ExternalFileAction;
import com.google.devtools.build.lib.skyframe.FileFunction;
import com.google.devtools.build.lib.skyframe.FileStateFunction;
import com.google.devtools.build.lib.skyframe.PrecomputedFunction;
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
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BazelModuleResolutionFunction}. */
@RunWith(JUnit4.class)
public class BazelModuleResolutionFunctionTest extends FoundationTestCase {

  private MemoizingEvaluator evaluator;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;
  private FakeRegistry.Factory registryFactory;

  @Before
  public void setup() throws Exception {
    differencer = new SequencedRecordingDifferencer();
    registryFactory = new FakeRegistry.Factory();
    evaluationContext =
        EvaluationContext.newBuilder().setNumThreads(8).setEventHandler(reporter).build();

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
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, rootDirectory, ImmutableMap.of()))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .put(SkyFunctions.BAZEL_MODULE_RESOLUTION, new BazelModuleResolutionFunction())
                .put(
                    SkyFunctions.CLIENT_ENVIRONMENT_VARIABLE,
                    new ClientEnvironmentFunction(new AtomicReference<>(ImmutableMap.of())))
                .buildOrThrow(),
            differencer);

    PrecomputedValue.STARLARK_SEMANTICS.set(
        differencer,
        StarlarkSemantics.builder().setBool(BuildLanguageOptions.ENABLE_BZLMOD, true).build());
    ModuleFileFunction.IGNORE_DEV_DEPS.set(differencer, false);
    ModuleFileFunction.MODULE_OVERRIDES.set(differencer, ImmutableMap.of());
    BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES.set(
        differencer, CheckDirectDepsMode.OFF);
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.ERROR);
    BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of());
  }

  @Test
  public void createValue_basic() throws Exception {
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
    ImmutableMap<String, ModuleOverride> overrides =
        ImmutableMap.of(
            "dep",
                MultipleVersionOverride.create(
                    ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), ""),
            "rules_java", LocalPathOverride.create("bleh"));

    BazelModuleResolutionValue value =
        BazelModuleResolutionFunction.createValue(depGraph, depGraph, overrides);
    assertThat(value.getCanonicalRepoNameLookup())
        .containsExactly(
            RepositoryName.MAIN,
            ModuleKey.ROOT,
            RepositoryName.create("dep~1.0"),
            createModuleKey("dep", "1.0"),
            RepositoryName.create("dep~2.0"),
            createModuleKey("dep", "2.0"),
            RepositoryName.create("rules_cc~1.0"),
            createModuleKey("rules_cc", "1.0"),
            RepositoryName.create("rules_java~override"),
            createModuleKey("rules_java", ""));
    assertThat(value.getModuleNameLookup())
        .containsExactly(
            "rules_cc", createModuleKey("rules_cc", "1.0"),
            "rules_java", createModuleKey("rules_java", ""));
    assertThat(value.getAbridgedModules())
        .containsExactlyElementsIn(
            depGraph.values().stream().map(AbridgedModule::from).collect(toImmutableList()));
  }

  private static ModuleExtensionUsage createModuleExtensionUsage(
      String bzlFile, String name, String... imports) {
    ImmutableBiMap.Builder<String, String> importsBuilder = ImmutableBiMap.builder();
    for (int i = 0; i < imports.length; i += 2) {
      importsBuilder.put(imports[i], imports[i + 1]);
    }
    return ModuleExtensionUsage.builder()
        .setExtensionBzlFile(bzlFile)
        .setExtensionName(name)
        .setImports(importsBuilder.buildOrThrow())
        .setLocation(Location.BUILTIN)
        .build();
  }

  @Test
  public void createValue_moduleExtensions() throws Exception {
    Module root =
        Module.builder()
            .setName("root")
            .setVersion(Version.parse("1.0"))
            .setKey(ModuleKey.ROOT)
            .addDep("rje", createModuleKey("rules_jvm_external", "1.0"))
            .addDep("rpy", createModuleKey("rules_python", "2.0"))
            .addExtensionUsage(
                createModuleExtensionUsage("@rje//:defs.bzl", "maven", "av", "autovalue"))
            .addExtensionUsage(
                createModuleExtensionUsage("@rpy//:defs.bzl", "pip", "numpy", "numpy"))
            .build();
    ModuleKey depKey = createModuleKey("dep", "2.0");
    Module dep =
        Module.builder()
            .setName("dep")
            .setVersion(Version.parse("2.0"))
            .setKey(depKey)
            .addDep("rules_python", createModuleKey("rules_python", "2.0"))
            .addExtensionUsage(
                createModuleExtensionUsage("@rules_python//:defs.bzl", "pip", "np", "numpy"))
            .addExtensionUsage(
                createModuleExtensionUsage("//:defs.bzl", "myext", "oneext", "myext"))
            .addExtensionUsage(
                createModuleExtensionUsage("//incredible:conflict.bzl", "myext", "twoext", "myext"))
            .build();
    ImmutableMap<ModuleKey, Module> depGraph = ImmutableMap.of(ModuleKey.ROOT, root, depKey, dep);

    ModuleExtensionId maven =
        ModuleExtensionId.create(
            Label.parseCanonical("@@rules_jvm_external~1.0//:defs.bzl"), "maven");
    ModuleExtensionId pip =
        ModuleExtensionId.create(Label.parseCanonical("@@rules_python~2.0//:defs.bzl"), "pip");
    ModuleExtensionId myext =
        ModuleExtensionId.create(Label.parseCanonical("@@dep~2.0//:defs.bzl"), "myext");
    ModuleExtensionId myext2 =
        ModuleExtensionId.create(
            Label.parseCanonical("@@dep~2.0//incredible:conflict.bzl"), "myext");

    BazelModuleResolutionValue value =
        BazelModuleResolutionFunction.createValue(depGraph, depGraph, ImmutableMap.of());
    assertThat(value.getExtensionUsagesTable()).hasSize(5);
    assertThat(value.getExtensionUsagesTable())
        .containsCell(maven, ModuleKey.ROOT, root.getExtensionUsages().get(0));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(pip, ModuleKey.ROOT, root.getExtensionUsages().get(1));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(pip, depKey, dep.getExtensionUsages().get(0));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(myext, depKey, dep.getExtensionUsages().get(1));
    assertThat(value.getExtensionUsagesTable())
        .containsCell(myext2, depKey, dep.getExtensionUsages().get(2));

    assertThat(value.getExtensionUniqueNames())
        .containsExactly(
            maven, "rules_jvm_external~1.0~maven",
            pip, "rules_python~2.0~pip",
            myext, "dep~2.0~myext",
            myext2, "dep~2.0~myext2");

    assertThat(value.getFullRepoMapping(ModuleKey.ROOT))
        .isEqualTo(
            createRepositoryMapping(
                ModuleKey.ROOT,
                "",
                "",
                "root",
                "",
                "rje",
                "rules_jvm_external~1.0",
                "rpy",
                "rules_python~2.0",
                "av",
                "rules_jvm_external~1.0~maven~autovalue",
                "numpy",
                "rules_python~2.0~pip~numpy"));
    assertThat(value.getFullRepoMapping(depKey))
        .isEqualTo(
            createRepositoryMapping(
                depKey,
                "dep",
                "dep~2.0",
                "rules_python",
                "rules_python~2.0",
                "np",
                "rules_python~2.0~pip~numpy",
                "oneext",
                "dep~2.0~myext~myext",
                "twoext",
                "dep~2.0~myext2~myext"));
  }

  @Test
  public void testBazelInvalidCompatibility() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0dd'])");

    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent("invalid version argument '>5.1.0dd'");
  }

  @Test
  public void testSimpleBazelCompatibilityFailure() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.4'])");

    embedBazelVersion("5.1.4");
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "Bazel version 5.1.4 is not compatible with module \"mod@1.0\" (bazel_compatibility:"
            + " [>5.1.0, <5.1.4])");
  }

  @Test
  public void testBazelCompatibilityWarning() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.4'])");

    embedBazelVersion("5.1.4");
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.WARNING);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isFalse();
    assertContainsEvent(
        "Bazel version 5.1.4 is not compatible with module \"mod@1.0\" (bazel_compatibility:"
            + " [>5.1.0, <5.1.4])");
  }

  @Test
  public void testDisablingBazelCompatibility() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.4'])");

    embedBazelVersion("5.1.4");
    BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE.set(
        differencer, BazelCompatibilityMode.OFF);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isFalse();
    assertDoesNotContainEvent(
        "Bazel version 5.1.4 is not compatible with module \"mod@1.0\" (bazel_compatibility:"
            + " [>5.1.0, <5.1.4])");
  }

  @Test
  public void testBazelCompatibilitySuccess() throws Exception {
    setupModulesForCompatibility();

    embedBazelVersion("5.1.4-pre.20220421.3");
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testBazelCompatibilityFailure() throws Exception {
    setupModulesForCompatibility();

    embedBazelVersion("5.1.5rc444");
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertContainsEvent(
        "Bazel version 5.1.5rc444 is not compatible with module \"b@1.0\" (bazel_compatibility:"
            + " [<=5.1.4, -5.1.2])");
  }

  private void embedBazelVersion(String version) {
    // Double-get version-info to determine if it's the cached instance or not, and if not cache it.
    BlazeVersionInfo blazeInfo1 = BlazeVersionInfo.instance();
    BlazeVersionInfo blazeInfo2 = BlazeVersionInfo.instance();
    if (blazeInfo1 != blazeInfo2) {
      BlazeVersionInfo.setBuildInfo(ImmutableMap.of());
      blazeInfo1 = BlazeVersionInfo.instance();
    }

    // embed new version
    Map<String, String> blazeInfo = blazeInfo1.getBuildData();
    blazeInfo.remove(BlazeVersionInfo.BUILD_LABEL);
    blazeInfo.put(BlazeVersionInfo.BUILD_LABEL, version);
  }

  private void setupModulesForCompatibility() throws IOException {
    /* Root depends on "a" which depends on "b"
       The only versions that would work with root, a and b compatibility constrains are between
       -not including- 5.1.2 and 5.1.4.
       Ex: 5.1.3rc44, 5.1.3, 5.1.4-pre22.44
    */
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0', bazel_compatibility=['>5.1.0', '<5.1.6'])",
        "bazel_dep(name = 'a', version = '1.0')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/foo")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0', bazel_compatibility=['>=5.1.2', '-5.1.4']);",
                "bazel_dep(name='b', version='1.0')")
            .addModule(
                createModuleKey("b", "1.0"),
                "module(name='b', version='1.0', bazel_compatibility=['<=5.1.4', '-5.1.2']);");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
  }

  @Test
  public void testYankedVersionCheckSuccess() throws Exception {
    setupModulesForYankedVersion();
    reporter.removeHandler(failFastHandler);
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);

    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString())
        .contains(
            "Yanked version detected in your resolved dependency graph: b@1.0, for the reason: 1.0"
                + " is a bad version!");
  }

  @Test
  public void testYankedVersionCheckIgnoredByAll() throws Exception {
    setupModulesForYankedVersion();
    BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS.set(differencer, ImmutableList.of("all"));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testYankedVersionCheckIgnoredBySpecific() throws Exception {
    setupModulesForYankedVersion();
    BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS.set(
        differencer, ImmutableList.of("b@1.0"));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isFalse();
  }

  @Test
  public void testBadYankedVersionFormat() throws Exception {
    setupModulesForYankedVersion();
    BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS.set(
        differencer, ImmutableList.of("b~1.0"));
    EvaluationResult<BazelModuleResolutionValue> result =
        evaluator.evaluate(ImmutableList.of(BazelModuleResolutionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().toString())
        .contains(
            "Parsing command line flag --allow_yanked_versions=b~1.0 failed, module versions must"
                + " be of the form '<module name>@<version>'");
  }

  private void setupModulesForYankedVersion() throws Exception {
    scratch.file(
        rootDirectory.getRelative("MODULE.bazel").getPathString(),
        "module(name='mod', version='1.0')",
        "bazel_dep(name = 'a', version = '1.0')");

    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/bar")
            .addModule(
                createModuleKey("a", "1.0"),
                "module(name='a', version='1.0');",
                "bazel_dep(name='b', version='1.0')")
            .addModule(createModuleKey("b", "1.0"), "module(name='b', version='1.0');")
            .addYankedVersion("b", ImmutableMap.of(Version.parse("1.0"), "1.0 is a bad version!"));
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));
  }
}
