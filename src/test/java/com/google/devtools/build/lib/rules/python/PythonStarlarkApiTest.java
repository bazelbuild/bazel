// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.python;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the ability of Starlark code to interact with the Python rules via the py provider. */
@RunWith(JUnit4.class)
public class PythonStarlarkApiTest extends BuildViewTestCase {

  /**
   * Defines userlib in //pkg:rules.bzl, which acts as a Starlark-defined version of py_library.
   *
   * <p>If {@code legacyProviderAllowed} is true, both the legacy and modern providers are returned.
   * Otherwise only the modern provider is returned. In both cases the modern provider of the
   * dependency is consumed while the native one is ignored.
   */
  private void defineUserlibRule(boolean legacyProviderAllowed) throws Exception {
    String returnExpr =
        legacyProviderAllowed ? "struct(py=legacy_info, providers=[modern_info])" : "[modern_info]";
    scratch.file(
        "pkg/rules.bzl",
        "def _userlib_impl(ctx):",
        "    dep_infos = [dep[PyInfo] for dep in ctx.attr.deps]",
        "    transitive_sources = depset(",
        "        direct=ctx.files.srcs,",
        "        transitive=[py.transitive_sources for py in dep_infos],",
        "        order='postorder')",
        "    uses_shared_libraries = \\",
        "        any([py.uses_shared_libraries for py in dep_infos]) or \\",
        "        ctx.attr.uses_shared_libraries",
        "    imports = depset(",
        "        direct=ctx.attr.imports,",
        "        transitive=[py.imports for py in dep_infos])",
        "    has_py2_only_sources = \\",
        "        any([py.has_py2_only_sources for py in dep_infos]) or \\",
        "        ctx.attr.has_py2_only_sources",
        "    has_py3_only_sources = \\",
        "        any([py.has_py3_only_sources for py in dep_infos]) or \\",
        "        ctx.attr.has_py3_only_sources",
        "    legacy_info = struct(",
        "        transitive_sources = transitive_sources,",
        "        uses_shared_libraries = uses_shared_libraries,",
        "        imports = imports,",
        "        has_py2_only_sources = has_py2_only_sources,",
        "        has_py3_only_sources = has_py3_only_sources)",
        "    modern_info = PyInfo(",
        "        transitive_sources = transitive_sources,",
        "        uses_shared_libraries = uses_shared_libraries,",
        "        imports = imports,",
        "        has_py2_only_sources = has_py2_only_sources,",
        "        has_py3_only_sources = has_py3_only_sources)",
        "    return " + returnExpr,
        "",
        "userlib = rule(",
        "    implementation = _userlib_impl,",
        "    attrs = {",
        "        'srcs': attr.label_list(allow_files=True),",
        "        'deps': attr.label_list(providers=[['py'], [PyInfo]]),",
        "        'uses_shared_libraries': attr.bool(),",
        "        'imports': attr.string_list(),",
        "        'has_py2_only_sources': attr.bool(),",
        "        'has_py3_only_sources': attr.bool(),",
        "    },",
        ")");
  }

  @Test
  public void librarySandwich_LegacyProviderAllowed() throws Exception {
    doLibrarySandwichTest(/*legacyProviderAllowed=*/ true);
  }

  @Test
  public void librarySandwich_LegacyProviderDisallowed() throws Exception {
    doLibrarySandwichTest(/*legacyProviderAllowed=*/ false);
  }

  private void doLibrarySandwichTest(boolean legacyProviderAllowed) throws Exception {
    useConfiguration(
        "--incompatible_disallow_legacy_py_provider=" + (legacyProviderAllowed ? "false" : "true"));
    defineUserlibRule(legacyProviderAllowed);
    scratch.file(
        "pkg/BUILD",
        "load(':rules.bzl', 'userlib')",
        "userlib(",
        "    name = 'loweruserlib',",
        "    srcs = ['loweruserlib.py'],",
        "    uses_shared_libraries = True,",
        "    imports = ['loweruserlib_path'],",
        "    has_py2_only_sources = True,",
        ")",
        "py_library(",
        "    name = 'pylib',",
        "    srcs = ['pylib.py'],",
        "    deps = [':loweruserlib'],",
        // No imports attribute here because Google-internal Python rules don't have this attribute.
        "    srcs_version = 'PY3ONLY'",
        ")",
        "userlib(",
        "    name = 'upperuserlib',",
        "    srcs = ['upperuserlib.py'],",
        "    deps = [':pylib'],",
        "    imports = ['upperuserlib_path'],",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//pkg:upperuserlib");

    if (legacyProviderAllowed) {
      StructImpl legacyInfo = PyProviderUtils.getLegacyProvider(target);
      assertThat(PyStructUtils.getTransitiveSources(legacyInfo).toList())
          .containsExactly(
              getSourceArtifact("pkg/loweruserlib.py"),
              getSourceArtifact("pkg/pylib.py"),
              getSourceArtifact("pkg/upperuserlib.py"));
      assertThat(PyStructUtils.getUsesSharedLibraries(legacyInfo)).isTrue();
      assertThat(PyStructUtils.getImports(legacyInfo).toList())
          .containsExactly("loweruserlib_path", "upperuserlib_path");
      assertThat(PyStructUtils.getHasPy2OnlySources(legacyInfo)).isTrue();
      assertThat(PyStructUtils.getHasPy3OnlySources(legacyInfo)).isTrue();
    }

    PyInfo modernInfo = PyProviderUtils.getModernProvider(target);
    assertThat(modernInfo.getTransitiveSources().toCollection(Artifact.class))
        .containsExactly(
            getSourceArtifact("pkg/loweruserlib.py"),
            getSourceArtifact("pkg/pylib.py"),
            getSourceArtifact("pkg/upperuserlib.py"));
    assertThat(modernInfo.getUsesSharedLibraries()).isTrue();
    assertThat(modernInfo.getImports().toCollection(String.class))
        .containsExactly("loweruserlib_path", "upperuserlib_path");
    assertThat(modernInfo.getHasPy2OnlySources()).isTrue();
    assertThat(modernInfo.getHasPy3OnlySources()).isTrue();
  }

  @Test
  public void runtimeSandwich() throws Exception {
    scratch.file(
        "pkg/rules.bzl",
        "def _userruntime_impl(ctx):",
        "    info = ctx.attr.runtime[PyRuntimeInfo]",
        "    return [PyRuntimeInfo(",
        "        interpreter = ctx.file.interpreter,",
        "        files = depset(direct = ctx.files.files, transitive=[info.files]),",
        "        python_version = info.python_version)]",
        "",
        "userruntime = rule(",
        "    implementation = _userruntime_impl,",
        "    attrs = {",
        "        'runtime': attr.label(),",
        "        'interpreter': attr.label(allow_single_file=True),",
        "        'files': attr.label_list(allow_files=True),",
        "    },",
        ")");
    scratch.file(
        "pkg/BUILD",
        "load('"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/python:toolchain.bzl', "
            + "'py_runtime_pair')",
        "load(':rules.bzl', 'userruntime')",
        "py_runtime(",
        "    name = 'pyruntime',",
        "    interpreter = ':intr',",
        "    files = ['data.txt'],",
        "    python_version = 'PY2',",
        ")",
        "userruntime(",
        "    name = 'userruntime',",
        "    runtime = ':pyruntime',",
        "    interpreter = ':userintr',",
        "    files = ['userdata.txt'],",
        ")",
        "py_runtime_pair(",
        "    name = 'userruntime_pair',",
        "    py2_runtime = 'userruntime',",
        ")",
        "toolchain(",
        "    name = 'usertoolchain',",
        "    toolchain = ':userruntime_pair',",
        "    toolchain_type = '"
            + TestConstants.TOOLS_REPOSITORY
            + "//tools/python:toolchain_type',",
        ")",
        "py_binary(",
        "    name = 'pybin',",
        "    srcs = ['pybin.py'],",
        ")");
    useConfiguration("--extra_toolchains=//pkg:usertoolchain");
    ConfiguredTarget target = getConfiguredTarget("//pkg:pybin");
    assertThat(collectRunfiles(target).toList())
        .containsAtLeast(getSourceArtifact("pkg/data.txt"), getSourceArtifact("pkg/userdata.txt"));
  }
}
