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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.bzlmod.BazelLockFileFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BazelModuleResolutionFunction;
import com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil;
import com.google.devtools.build.lib.bazel.bzlmod.FakeRegistry;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.BazelCompatibilityMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.CheckDirectDepsMode;
import com.google.devtools.build.lib.bazel.repository.RepositoryOptions.LockfileMode;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Injected;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.OriginKey;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderNameGroup;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.util.NoSuchElementException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class StarlarkDocExtractTest extends BuildViewTestCase {
  private Path moduleRoot; // initialized by extraPrecomputedValues
  private FakeRegistry registry;

  @Override
  protected ImmutableList<Injected> extraPrecomputedValues() {
    // TODO(b/285924565): support --enable_bzlmod in BuildViewTestCase tests without needing the
    // boilerplate below.
    try {
      moduleRoot = scratch.dir("modules");
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
    registry = FakeRegistry.DEFAULT_FACTORY.newFakeRegistry(moduleRoot.getPathString());
    return ImmutableList.of(
        PrecomputedValue.injected(
            ModuleFileFunction.REGISTRIES, ImmutableList.of(registry.getUrl())),
        PrecomputedValue.injected(ModuleFileFunction.IGNORE_DEV_DEPS, false),
        PrecomputedValue.injected(ModuleFileFunction.MODULE_OVERRIDES, ImmutableMap.of()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.ALLOWED_YANKED_VERSIONS, ImmutableList.of()),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.CHECK_DIRECT_DEPENDENCIES, CheckDirectDepsMode.WARNING),
        PrecomputedValue.injected(
            BazelModuleResolutionFunction.BAZEL_COMPATIBILITY_MODE, BazelCompatibilityMode.ERROR),
        PrecomputedValue.injected(BazelLockFileFunction.LOCKFILE_MODE, LockfileMode.OFF));
  }

  private static ModuleInfo protoFromBinaryFileWriteAction(Action action) throws Exception {
    assertThat(action).isInstanceOf(BinaryFileWriteAction.class);
    return ModuleInfo.parseFrom(
        ((BinaryFileWriteAction) action).getSource().openStream(),
        ExtensionRegistry.getEmptyRegistry());
  }

  private static ModuleInfo protoFromTextFileWriteAction(Action action) throws Exception {
    assertThat(action).isInstanceOf(FileWriteAction.class);
    return TextFormat.parse(
        ((FileWriteAction) action).getFileContents(),
        ExtensionRegistry.getEmptyRegistry(),
        ModuleInfo.class);
  }

  @Test
  public void basicFunctionality() throws Exception {
    useConfiguration("--experimental_enable_starlark_doc_extract");
    scratch.file(
        "foo.bzl", //
        "'''Module doc string'''",
        "True");
    scratch.file(
        "BUILD", //
        "starlark_doc_extract(",
        "    name = 'extract',",
        "    src = 'foo.bzl',",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//:extract");
    ModuleInfo moduleInfo =
        protoFromBinaryFileWriteAction(getGeneratingAction(target, "extract.binaryproto"));
    assertThat(moduleInfo.getModuleDocstring()).isEqualTo("Module doc string");
    assertThat(moduleInfo.getFile()).isEqualTo("//:foo.bzl");
  }

  @Test
  public void textprotoOut() throws Exception {
    useConfiguration("--experimental_enable_starlark_doc_extract");
    scratch.file(
        "foo.bzl", //
        "'''Module doc string'''",
        "True");
    scratch.file(
        "BUILD", //
        "starlark_doc_extract(",
        "    name = 'extract',",
        "    src = 'foo.bzl',",
        ")");
    ConfiguredTarget ruleTarget = getConfiguredTarget("//:extract");
    // Verify that we do not generate textproto output unless explicitly requested.
    assertThrows(
        NoSuchElementException.class, () -> getGeneratingAction(ruleTarget, "extract.textproto"));

    ConfiguredTarget textprotoOutputTarget = getConfiguredTarget("//:extract.textproto");
    ModuleInfo moduleInfo =
        protoFromTextFileWriteAction(
            getGeneratingAction(textprotoOutputTarget, "extract.textproto"));
    assertThat(moduleInfo.getModuleDocstring()).isEqualTo("Module doc string");
  }

  @Test
  public void symbolNames() throws Exception {
    useConfiguration("--experimental_enable_starlark_doc_extract");
    scratch.file(
        "foo.bzl", //
        "def func1():",
        "    pass",
        "def func2():",
        "    pass",
        "def _hidden():",
        "    pass");
    scratch.file(
        "BUILD", //
        "starlark_doc_extract(",
        "    name = 'extract_some',",
        "    src = 'foo.bzl',",
        "    symbol_names = ['func1'],",
        ")",
        "starlark_doc_extract(",
        "    name = 'extract_all',",
        "    src = 'foo.bzl',",
        ")");

    ModuleInfo dumpSome =
        protoFromBinaryFileWriteAction(
            getGeneratingAction(
                getConfiguredTarget("//:extract_some"), "extract_some.binaryproto"));
    assertThat(dumpSome.getFuncInfoList().stream().map(StarlarkFunctionInfo::getFunctionName))
        .containsExactly("func1");

    ModuleInfo dumpAll =
        protoFromBinaryFileWriteAction(
            getGeneratingAction(getConfiguredTarget("//:extract_all"), "extract_all.binaryproto"));
    assertThat(dumpAll.getFuncInfoList().stream().map(StarlarkFunctionInfo::getFunctionName))
        .containsExactly("func1", "func2");
  }

  @Test
  public void originKey() throws Exception {
    useConfiguration("--experimental_enable_starlark_doc_extract");
    scratch.file(
        "origin.bzl", //
        "def my_macro():",
        "    pass",
        "MyInfo = provider()",
        "MyOtherInfo = provider()",
        "my_rule = rule(",
        "    implementation = lambda ctx: None,",
        "    attrs = {'a': attr.label(providers = [MyInfo, MyOtherInfo])},",
        "    provides = [MyInfo, MyOtherInfo],",
        ")",
        "my_aspect = aspect(implementation = lambda target, ctx: None)");
    scratch.file(
        "renamer.bzl", //
        "load(':origin.bzl', 'my_macro', 'MyInfo', 'MyOtherInfo', 'my_rule', 'my_aspect')",
        "namespace = struct(",
        "    renamed_macro = my_macro,",
        "    RenamedInfo = MyInfo,",
        "    renamed_rule = my_rule,",
        "    renamed_aspect = my_aspect,",
        ")",
        "other_namespace = struct(",
        "    RenamedOtherInfo = MyOtherInfo,",
        ")");
    scratch.file(
        "BUILD", //
        "starlark_doc_extract(",
        "    name = 'extract_renamed',",
        "    src = 'renamer.bzl',",
        "    symbol_names = ['namespace'],",
        ")");

    ModuleInfo moduleInfo =
        protoFromBinaryFileWriteAction(
            getGeneratingAction(
                getConfiguredTarget("//:extract_renamed"), "extract_renamed.binaryproto"));

    assertThat(moduleInfo.getFuncInfoList().stream().map(StarlarkFunctionInfo::getFunctionName))
        .containsExactly("namespace.renamed_macro");
    assertThat(moduleInfo.getFuncInfoList().stream().map(StarlarkFunctionInfo::getOriginKey))
        .containsExactly(
            OriginKey.newBuilder().setName("my_macro").setFile("//:origin.bzl").build());

    assertThat(moduleInfo.getProviderInfoList().stream().map(ProviderInfo::getProviderName))
        .containsExactly("namespace.RenamedInfo");
    assertThat(moduleInfo.getProviderInfoList().stream().map(ProviderInfo::getOriginKey))
        .containsExactly(OriginKey.newBuilder().setName("MyInfo").setFile("//:origin.bzl").build());

    assertThat(moduleInfo.getRuleInfoList().stream().map(RuleInfo::getRuleName))
        .containsExactly("namespace.renamed_rule");
    assertThat(moduleInfo.getRuleInfoList().stream().map(RuleInfo::getOriginKey))
        .containsExactly(
            OriginKey.newBuilder().setName("my_rule").setFile("//:origin.bzl").build());

    assertThat(moduleInfo.getRuleInfo(0).getAttributeList())
        .containsExactly(
            ModuleInfoExtractor.IMPLICIT_NAME_ATTRIBUTE_INFO,
            AttributeInfo.newBuilder()
                .setName("a")
                .setType(AttributeType.LABEL)
                .setDefaultValue("None")
                .addProviderNameGroup(
                    ProviderNameGroup.newBuilder()
                        .addProviderName("namespace.RenamedInfo")
                        .addProviderName("other_namespace.RenamedOtherInfo")
                        .addOriginKey(
                            OriginKey.newBuilder().setName("MyInfo").setFile("//:origin.bzl"))
                        .addOriginKey(
                            OriginKey.newBuilder().setName("MyOtherInfo").setFile("//:origin.bzl")))
                .build());
    assertThat(moduleInfo.getRuleInfo(0).getAdvertisedProviders())
        .isEqualTo(
            ProviderNameGroup.newBuilder()
                .addProviderName("namespace.RenamedInfo")
                .addProviderName("other_namespace.RenamedOtherInfo")
                .addOriginKey(OriginKey.newBuilder().setName("MyInfo").setFile("//:origin.bzl"))
                .addOriginKey(
                    OriginKey.newBuilder().setName("MyOtherInfo").setFile("//:origin.bzl"))
                .build());

    assertThat(moduleInfo.getAspectInfoList().stream().map(AspectInfo::getAspectName))
        .containsExactly("namespace.renamed_aspect");
    assertThat(moduleInfo.getAspectInfoList().stream().map(AspectInfo::getOriginKey))
        .containsExactly(
            OriginKey.newBuilder().setName("my_aspect").setFile("//:origin.bzl").build());
  }

  @Test
  public void originKeyFileAndModuleInfoFileLabels_forBzlFileInBzlmodModule_areDisplayForm()
      throws Exception {
    setBuildLanguageOptions("--enable_bzlmod");
    useConfiguration("--experimental_enable_starlark_doc_extract");
    scratch.overwriteFile("MODULE.bazel", "bazel_dep(name='origin_repo', version='0.1')");
    registry.addModule(
        BzlmodTestUtil.createModuleKey("origin_repo", "0.1"),
        "module(name='origin_repo', version='0.1')");
    Path originRepoPath = moduleRoot.getRelative("origin_repo~0.1");
    scratch.file(originRepoPath.getRelative("WORKSPACE").getPathString());
    scratch.file(
        originRepoPath.getRelative("BUILD").getPathString(), //
        "exports_files(['origin.bzl'])");
    scratch.file(
        originRepoPath.getRelative("origin.bzl").getPathString(), //
        "def my_macro():",
        "    pass",
        "MyInfo = provider()",
        "my_rule = rule(",
        "    implementation = lambda ctx: None,",
        "    attrs = {'a': attr.label(providers = [MyInfo])},",
        "    provides = [MyInfo],",
        ")",
        "my_aspect = aspect(implementation = lambda target, ctx: None)");
    scratch.file(
        "renamer.bzl", //
        "load('@origin_repo//:origin.bzl', 'my_macro', 'MyInfo', 'my_rule', 'my_aspect')",
        "namespace = struct(",
        "    renamed_macro = my_macro,",
        "    RenamedInfo = MyInfo,",
        "    renamed_rule = my_rule,",
        "    renamed_aspect = my_aspect,",
        ")");
    scratch.file(
        "BUILD", //
        "starlark_doc_extract(",
        "    name = 'extract_origin',",
        "    src = '@origin_repo//:origin.bzl',",
        ")",
        "starlark_doc_extract(",
        "    name = 'extract_renamed',",
        "    src = 'renamer.bzl',",
        ")");

    // verify that ModuleInfo.name for a .bzl file in another bzlmod module is in display form, i.e.
    // "@origin_repo//:origin.bzl" as opposed to "@@origin_repo~0.1//:origin.bzl"
    ModuleInfo originModuleInfo =
        protoFromBinaryFileWriteAction(
            getGeneratingAction(
                getConfiguredTarget("//:extract_origin"), "extract_origin.binaryproto"));
    assertThat(originModuleInfo.getFile()).isEqualTo("@origin_repo//:origin.bzl");

    // verify that OriginKey.name for entities defined in a .bzl file in another bzlmod module is in
    // display form, i.e. "@origin_repo//:origin.bzl" as opposed to "@@origin_repo~0.1//:origin.bzl"
    ModuleInfo renamedModuleInfo =
        protoFromBinaryFileWriteAction(
            getGeneratingAction(
                getConfiguredTarget("//:extract_renamed"), "extract_renamed.binaryproto"));
    assertThat(renamedModuleInfo.getFile()).isEqualTo("//:renamer.bzl");
    assertThat(renamedModuleInfo.getFuncInfo(0).getOriginKey().getFile())
        .isEqualTo("@origin_repo//:origin.bzl");
    assertThat(renamedModuleInfo.getProviderInfo(0).getOriginKey().getFile())
        .isEqualTo("@origin_repo//:origin.bzl");
    assertThat(renamedModuleInfo.getAspectInfo(0).getOriginKey().getFile())
        .isEqualTo("@origin_repo//:origin.bzl");
    assertThat(renamedModuleInfo.getRuleInfo(0).getOriginKey().getFile())
        .isEqualTo("@origin_repo//:origin.bzl");
    assertThat(
            renamedModuleInfo
                .getRuleInfo(0)
                .getAttribute(1) // 0 is the implicit name attribute
                .getProviderNameGroup(0)
                .getOriginKey(0)
                .getFile())
        .isEqualTo("@origin_repo//:origin.bzl");
    assertThat(renamedModuleInfo.getRuleInfo(0).getAdvertisedProviders().getOriginKey(0).getFile())
        .isEqualTo("@origin_repo//:origin.bzl");
  }

  @Test
  public void exportNestedFunctionsAndLambdas() throws Exception {
    useConfiguration("--experimental_enable_starlark_doc_extract");
    scratch.file(
        "origin.bzl", //
        "def return_nested():",
        "    def nested(x):",
        "        '''My nested function'''",
        "        pass",
        "    return nested",
        "",
        "def return_lambda():",
        "    return lambda y: y");
    scratch.file(
        "exporter.bzl", //
        "load(':origin.bzl', 'return_nested', 'return_lambda')",
        "exported_nested = return_nested()",
        "exported_lambda = return_lambda()");
    scratch.file(
        "BUILD", //
        "starlark_doc_extract(",
        "    name = 'extract_exporter',",
        "    src = 'exporter.bzl',",
        ")");

    ModuleInfo moduleInfo =
        protoFromBinaryFileWriteAction(
            getGeneratingAction(
                getConfiguredTarget("//:extract_exporter"), "extract_exporter.binaryproto"));

    assertThat(moduleInfo.getFuncInfoList())
        .containsExactly(
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("exported_nested")
                .setDocString("My nested function")
                .addParameter(FunctionParamInfo.newBuilder().setName("x").setMandatory(true))
                .setOriginKey(
                    // OriginKey.name for nested functions is explicitly unset
                    OriginKey.newBuilder().setFile("//:origin.bzl"))
                .build(),
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("exported_lambda")
                .addParameter(FunctionParamInfo.newBuilder().setName("y").setMandatory(true))
                .setOriginKey(
                    // OriginKey.name for lambdas is explicitly unset
                    OriginKey.newBuilder().setFile("//:origin.bzl"))
                .build());
  }
}
