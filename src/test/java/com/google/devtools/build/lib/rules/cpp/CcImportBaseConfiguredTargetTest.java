// Copyright 2015 The Bazel Authors. All rights reserved.
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
// Copyright 2006 Google Inc. All rights reserved.

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** "White-box" unit test of cc_import rule. */
@RunWith(JUnit4.class)
public abstract class CcImportBaseConfiguredTargetTest extends BuildViewTestCase {
  protected String starlarkImplementationLoadStatement = "";

  @Before
  public void setStarlarkImplementationLoadStatement() throws Exception {
    setBuildLanguageOptions(StarlarkCcCommonTestHelper.CC_STARLARK_WHITELIST_FLAG);
    invalidatePackages();
    setIsStarlarkImplementation();
  }

  protected abstract void setIsStarlarkImplementation();

  @Test
  public void testCcImportRule() throws Exception {
    scratch.file(
        "third_party/BUILD",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'a_import',",
        "  static_library = 'A.a',",
        "  shared_library = 'A.so',",
        "  interface_library = 'A.ifso',",
        "  hdrs = ['a.h'],",
        "  alwayslink = 1,",
        "  system_provided = 0,",
        ")");
    getConfiguredTarget("//third_party:a_import");
  }

  @Test
  public void testWrongCcImportDefinitions() throws Exception {
    checkError(
        "a",
        "foo",
        "does not produce any cc_import static_library files " + "(expected .a, .lib or .pic.a)",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'foo',",
        "  static_library = 'libfoo.so',",
        ")");
    checkError(
        "b",
        "foo",
        "does not produce any cc_import shared_library files (expected .so, .dylib or .dll)",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'foo',",
        "  shared_library = 'libfoo.a',",
        ")");
    checkError(
        "c",
        "foo",
        "does not produce any cc_import interface_library files "
            + "(expected .ifso, .tbd, .lib, .so or .dylib)",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'foo',",
        "  shared_library = 'libfoo.dll',",
        "  interface_library = 'libfoo.a',",
        ")");
    checkError(
        "d",
        "foo",
        "'shared_library' shouldn't be specified when 'system_provided' is true",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'foo',",
        "  shared_library = 'libfoo.so',",
        "  system_provided = 1,",
        ")");
    checkError(
        "e",
        "foo",
        "'shared_library' should be specified when 'system_provided' is false",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'foo',",
        "  interface_library = 'libfoo.ifso',",
        "  system_provided = 0,",
        ")");
  }

  @Test
  public void testRuntimeOnlyCcImportDefinitionsOnWindows() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder()
                .withFeatures(
                    CppRuleClasses.COPY_DYNAMIC_LIBRARIES_TO_BINARY,
                    CppRuleClasses.TARGETS_WINDOWS));
    useConfiguration();
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', shared_library = 'libfoo.dll')");
    Artifact dynamicLibrary =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getResolvedSymlinkDynamicLibrary();
    Iterable<Artifact> dynamicLibrariesForRuntime =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getDynamicLibrariesForRuntime(/* linkingStatically= */ false);
    assertThat(dynamicLibrary).isEqualTo(null);
    assertThat(artifactsToStrings(dynamicLibrariesForRuntime)).containsExactly("src a/libfoo.dll");
  }

  @Test
  public void testCcImportWithStaticLibrary() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', static_library = 'libfoo.a')");
    Artifact library =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getStaticLibrary();
    assertThat(artifactsToStrings(ImmutableList.of(library))).containsExactly("src a/libfoo.a");
  }

  @Test
  public void testCcImportWithSharedLibrary() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', shared_library = 'libfoo.so')");
    Artifact dynamicLibrary =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getResolvedSymlinkDynamicLibrary();
    Iterable<Artifact> dynamicLibrariesForRuntime =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getDynamicLibrariesForRuntime(/* linkingStatically= */ false);
    assertThat(artifactsToStrings(ImmutableList.of(dynamicLibrary)))
        .containsExactly("src a/libfoo.so");
    assertThat(artifactsToStrings(dynamicLibrariesForRuntime))
        .containsExactly("bin _solib_k8/_U_S_Sa_Cfoo___Ua/libfoo.so");
  }

  @Test
  public void testCcImportWithVersionedSharedLibrary() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', shared_library = 'libfoo.so.1ab2.1_a2')");
    Artifact dynamicLibrary =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getResolvedSymlinkDynamicLibrary();
    Iterable<Artifact> dynamicLibrariesForRuntime =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getDynamicLibrariesForRuntime(/* linkingStatically= */ false);
    assertThat(artifactsToStrings(ImmutableList.of(dynamicLibrary)))
        .containsExactly("src a/libfoo.so.1ab2.1_a2");
    assertThat(artifactsToStrings(dynamicLibrariesForRuntime))
        .containsExactly("bin _solib_k8/_U_S_Sa_Cfoo___Ua/libfoo.so.1ab2.1_a2");
  }

  @Test
  public void testCcImportWithInvalidVersionedSharedLibrary() throws Exception {
    checkError(
        "a",
        "foo",
        "does not produce any cc_import shared_library files " + "(expected .so, .dylib or .dll)",
        starlarkImplementationLoadStatement,
        "cc_import(",
        "  name = 'foo',",
        "  shared_library = 'libfoo.so.1ab2.ab',",
        ")");
  }

  @Test
  public void testCcImportWithInterfaceSharedLibrary() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "b",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', shared_library = 'libfoo.so',"
                + " interface_library = 'libfoo.ifso')");
    ;
    Artifact library =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getResolvedSymlinkInterfaceLibrary();
    assertThat(artifactsToStrings(ImmutableList.of(library))).containsExactly("src b/libfoo.ifso");
    Iterable<Artifact> dynamicLibrariesForRuntime =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getDynamicLibrariesForRuntime(/* linkingStatically= */ false);
    assertThat(artifactsToStrings(dynamicLibrariesForRuntime))
        .containsExactly("bin _solib_k8/_U_S_Sb_Cfoo___Ub/libfoo.so");
  }

  @Test
  public void testCcImportWithBothStaticAndSharedLibraries() throws Exception {
    useConfiguration("--cpu=k8");
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', static_library = 'libfoo.a', shared_library = 'libfoo.so')");

    Artifact library =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getStaticLibrary();
    assertThat(artifactsToStrings(ImmutableList.of(library))).containsExactly("src a/libfoo.a");

    Artifact dynamicLibrary =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getResolvedSymlinkDynamicLibrary();
    Iterable<Artifact> dynamicLibrariesForRuntime =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getDynamicLibrariesForRuntime(/* linkingStatically= */ false);
    assertThat(artifactsToStrings(ImmutableList.of(dynamicLibrary)))
        .containsExactly("src a/libfoo.so");
    assertThat(artifactsToStrings(dynamicLibrariesForRuntime))
        .containsExactly("bin _solib_k8/_U_S_Sa_Cfoo___Ua/libfoo.so");
  }

  @Test
  public void testCcImportWithAlwaysLinkStaticLibrary() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', static_library = 'libfoo.a', alwayslink = 1)");
    boolean alwayslink =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getAlwayslink();
    assertThat(alwayslink).isTrue();
  }

  @Test
  public void testCcImportSystemProvidedIsTrue() throws Exception {
    ConfiguredTarget target =
        scratchConfiguredTarget(
            "a",
            "foo",
            starlarkImplementationLoadStatement,
            "cc_import(name = 'foo', interface_library = 'libfoo.ifso', system_provided = 1)");
    Artifact library =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getLibraries()
            .getSingleton()
            .getResolvedSymlinkInterfaceLibrary();
    assertThat(artifactsToStrings(ImmutableList.of(library))).containsExactly("src a/libfoo.ifso");
    Iterable<Artifact> dynamicLibrariesForRuntime =
        target
            .get(CcInfo.PROVIDER)
            .getCcLinkingContext()
            .getDynamicLibrariesForRuntime(/* linkingStatically= */ false);
    assertThat(artifactsToStrings(dynamicLibrariesForRuntime)).isEmpty();
  }

  @Test
  public void testCcImportProvideHeaderFiles() throws Exception {
    NestedSet<Artifact> headers =
        scratchConfiguredTarget(
                "a",
                "foo",
                starlarkImplementationLoadStatement,
                "cc_import(name = 'foo', static_library = 'libfoo.a', hdrs = ['foo.h'])")
            .get(CcInfo.PROVIDER)
            .getCcCompilationContext()
            .getDeclaredIncludeSrcs();
    assertThat(artifactsToStrings(headers)).containsExactly("src a/foo.h");
  }

  @Test
  public void testCcImportLoadedThroughMacro() throws Exception {
    setupTestCcImportLoadedThroughMacro(/* loadMacro= */ true);
    assertThat(getConfiguredTarget("//a:a")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void testCcImportNotLoadedThroughMacro() throws Exception {
    setupTestCcImportLoadedThroughMacro(/* loadMacro= */ false);
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("rules are deprecated");
  }

  private void setupTestCcImportLoadedThroughMacro(boolean loadMacro) throws Exception {
    useConfiguration("--incompatible_load_cc_rules_from_bzl");
    scratch.file(
        "a/BUILD",
        getAnalysisMock().ccSupport().getMacroLoadStatement(loadMacro, "cc_import"),
        "cc_import(name='a', static_library='a.a')");
  }
}
