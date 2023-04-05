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
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import java.util.NoSuchElementException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class StarlarkDocExtractTest extends BuildViewTestCase {

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
}
