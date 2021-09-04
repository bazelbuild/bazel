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

import static com.google.common.truth.Truth8.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static org.junit.Assert.fail;

import com.google.auto.value.AutoValue;
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
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BzlmodRepoRuleHelperImpl}. */
@RunWith(JUnit4.class)
public final class BzlmodRepoRuleHelperTest extends FoundationTestCase {

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
                .put(
                    SkyFunctions.MODULE_FILE,
                    new ModuleFileFunction(registryFactory, workspaceRoot))
                .put(SkyFunctions.DISCOVERY, new DiscoveryFunction())
                .put(SkyFunctions.SELECTION, new SelectionFunction())
                .put(
                    GET_REPO_SPEC_BY_NAME_FUNCTION,
                    new GetRepoSpecByNameFunction(new BzlmodRepoRuleHelperImpl()))
                .put(SkyFunctions.PRECOMPUTED, new PrecomputedFunction())
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);

    PrecomputedValue.STARLARK_SEMANTICS.set(differencer, StarlarkSemantics.DEFAULT);
  }

  @Test
  public void getRepoSpec_bazelModule() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/usr/local/modules")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='C',version='2.0')")
            .addModule(createModuleKey("C", "2.0"), "module(name='C', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<GetRepoSpecByNameValue> result =
        driver.evaluate(ImmutableList.of(getRepoSpecByNameKey("C.2.0")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    Optional<RepoSpec> repoSpec = result.get(getRepoSpecByNameKey("C.2.0")).rule();
    assertThat(repoSpec)
        .hasValue(
            RepoSpec.builder()
                .setRuleClassName("local_repository")
                .setAttributes(ImmutableMap.of("name", "C.2.0", "path", "/usr/local/modules/C.2.0"))
                .build());
  }

  @Test
  public void getRepoSpec_nonRegistryOverride() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "local_path_override(module_name='C',path='/foo/bar/C')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/usr/local/modules")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='C',version='2.0')")
            .addModule(createModuleKey("C", "2.0"), "module(name='C', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<GetRepoSpecByNameValue> result =
        driver.evaluate(ImmutableList.of(getRepoSpecByNameKey("C.")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    Optional<RepoSpec> repoSpec = result.get(getRepoSpecByNameKey("C.")).rule();
    assertThat(repoSpec)
        .hasValue(
            RepoSpec.builder()
                .setRuleClassName("local_repository")
                .setAttributes(
                    ImmutableMap.of(
                        "name", "C.",
                        "path", "/foo/bar/C"))
                .build());
  }

  @Test
  public void getRepoSpec_singleVersionOverride() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "single_version_override(",
        "  module_name='C',version='3.0',patches=['//:foo.patch'],patch_strip=1)");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/usr/local/modules")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='C',version='2.0')")
            .addModule(createModuleKey("C", "2.0"), "module(name='C', version='2.0')")
            .addModule(createModuleKey("C", "3.0"), "module(name='C', version='3.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<GetRepoSpecByNameValue> result =
        driver.evaluate(ImmutableList.of(getRepoSpecByNameKey("C.3.0")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    Optional<RepoSpec> repoSpec = result.get(getRepoSpecByNameKey("C.3.0")).rule();
    assertThat(repoSpec)
        .hasValue(
            RepoSpec.builder()
                // This obviously wouldn't work in the real world since local_repository doesn't
                // support patches -- but in the real world, registries also don't use
                // local_repository.
                .setRuleClassName("local_repository")
                .setAttributes(
                    ImmutableMap.of(
                        "name",
                        "C.3.0",
                        "path",
                        "/usr/local/modules/C.3.0",
                        "patches",
                        ImmutableList.of("//:foo.patch"),
                        "patch_args",
                        ImmutableList.of("-p1")))
                .build());
  }

  @Test
  public void getRepoSpec_multipleVersionOverride() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')",
        "bazel_dep(name='C',version='2.0')",
        "multiple_version_override(module_name='D',versions=['1.0','2.0'])");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/usr/local/modules")
            .addModule(
                createModuleKey("B", "1.0"),
                "module(name='B', version='1.0');bazel_dep(name='D',version='1.0')")
            .addModule(
                createModuleKey("C", "2.0"),
                "module(name='C', version='2.0');bazel_dep(name='D',version='2.0')")
            .addModule(createModuleKey("D", "1.0"), "module(name='D', version='1.0')")
            .addModule(createModuleKey("D", "2.0"), "module(name='D', version='2.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<GetRepoSpecByNameValue> result =
        driver.evaluate(ImmutableList.of(getRepoSpecByNameKey("D.2.0")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    Optional<RepoSpec> repoSpec = result.get(getRepoSpecByNameKey("D.2.0")).rule();
    assertThat(repoSpec)
        .hasValue(
            RepoSpec.builder()
                .setRuleClassName("local_repository")
                .setAttributes(ImmutableMap.of("name", "D.2.0", "path", "/usr/local/modules/D.2.0"))
                .build());
  }

  @Test
  public void getRepoSpec_notFound() throws Exception {
    scratch.file(
        workspaceRoot.getRelative("MODULE.bazel").getPathString(),
        "module(name='A',version='0.1')",
        "bazel_dep(name='B',version='1.0')");
    FakeRegistry registry =
        registryFactory
            .newFakeRegistry("/usr/local/modules")
            .addModule(createModuleKey("B", "1.0"), "module(name='B', version='1.0')");
    ModuleFileFunction.REGISTRIES.set(differencer, ImmutableList.of(registry.getUrl()));

    EvaluationResult<GetRepoSpecByNameValue> result =
        driver.evaluate(ImmutableList.of(getRepoSpecByNameKey("C")), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }

    Optional<RepoSpec> repoSpec = result.get(getRepoSpecByNameKey("C")).rule();
    assertThat(repoSpec).isEmpty();
  }

  /** A helper SkyFunction to invoke BzlmodRepoRuleHelper */
  private static final SkyFunctionName GET_REPO_SPEC_BY_NAME_FUNCTION =
      SkyFunctionName.createHermetic("GET_REPO_SPEC_BY_NAME_FUNCTION");

  @AutoValue
  abstract static class GetRepoSpecByNameValue implements SkyValue {
    abstract Optional<RepoSpec> rule();

    static GetRepoSpecByNameValue create(Optional<RepoSpec> rule) {
      return new AutoValue_BzlmodRepoRuleHelperTest_GetRepoSpecByNameValue(rule);
    }
  }

  private static final class GetRepoSpecByNameFunction implements SkyFunction {

    private final BzlmodRepoRuleHelper bzlmodRepoRuleHelper;

    public GetRepoSpecByNameFunction(BzlmodRepoRuleHelper bzlmodRepoRuleHelper) {
      this.bzlmodRepoRuleHelper = bzlmodRepoRuleHelper;
    }

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      String repositoryName = (String) skyKey.argument();
      Optional<RepoSpec> result;
      try {
        result = bzlmodRepoRuleHelper.getRepoSpec(env, repositoryName);
        if (env.valuesMissing()) {
          return null;
        }
      } catch (IOException e) {
        throw new GetRepoSpecByNameFunctionException(e, Transience.PERSISTENT);
      }
      return GetRepoSpecByNameValue.create(result);
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static final class Key extends AbstractSkyKey<String> {
    private Key(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return GET_REPO_SPEC_BY_NAME_FUNCTION;
    }
  }

  private static final class GetRepoSpecByNameFunctionException extends SkyFunctionException {
    GetRepoSpecByNameFunctionException(IOException e, Transience transience) {
      super(e, transience);
    }
  }

  private static SkyKey getRepoSpecByNameKey(String repositoryName) {
    return new Key(repositoryName);
  }
}
