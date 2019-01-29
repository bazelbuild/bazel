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
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the ability of Starlark code to interact with the Python rules via the py provider. */
@RunWith(JUnit4.class)
public class PythonStarlarkApiTest extends BuildViewTestCase {

  // TODO(#7054): Imports aren't propagated correctly. When we fix that, add it to the fields
  // tested here.

  /** Defines userlib in //pkg:rules.bzl, which acts as a Starlark-defined version of py_library. */
  private void defineUserlibRule() throws Exception {
    scratch.file(
        "pkg/rules.bzl",
        "def _userlib_impl(ctx):",
        "    dep_infos = [dep.py for dep in ctx.attr.deps]",
        "    transitive_sources = depset(",
        "        direct=ctx.files.srcs,",
        "        transitive=[py.transitive_sources for py in dep_infos],",
        "        order='postorder')",
        "    uses_shared_libraries = \\",
        "        any([py.uses_shared_libraries for py in dep_infos]) or \\",
        "        ctx.attr.uses_shared_libraries",
        "    has_py2_only_sources = \\",
        "        any([py.has_py2_only_sources for py in dep_infos]) or \\",
        "        ctx.attr.has_py2_only_sources",
        "    has_py3_only_sources = \\",
        "        any([py.has_py3_only_sources for py in dep_infos]) or \\",
        "        ctx.attr.has_py3_only_sources",
        "    legacy_info = struct(",
        "        transitive_sources = transitive_sources,",
        "        uses_shared_libraries = uses_shared_libraries,",
        "        has_py2_only_sources = has_py2_only_sources,",
        "        has_py3_only_sources = has_py3_only_sources)",
        "    modern_info = PyInfo(",
        "        transitive_sources = transitive_sources,",
        "        uses_shared_libraries = uses_shared_libraries,",
        "        has_py2_only_sources = has_py2_only_sources,",
        "        has_py3_only_sources = has_py3_only_sources)",
        "    return struct(py=legacy_info, providers=[modern_info])",
        "",
        "userlib = rule(",
        "    implementation = _userlib_impl,",
        "    attrs = {",
        "        'srcs': attr.label_list(allow_files=True),",
        "        'deps': attr.label_list(providers=['py']),",
        "        'uses_shared_libraries': attr.bool(),",
        "        'has_py2_only_sources': attr.bool(),",
        "        'has_py3_only_sources': attr.bool(),",
        "    },",
        ")");
  }

  @Test
  public void librarySandwich() throws Exception {
    // Use new version semantics so we don't validate source versions in py_library.
    useConfiguration("--experimental_allow_python_version_transitions=true");
    defineUserlibRule();
    scratch.file(
        "pkg/BUILD",
        "load(':rules.bzl', 'userlib')",
        "userlib(",
        "    name = 'loweruserlib',",
        "    srcs = ['loweruserlib.py'],",
        "    uses_shared_libraries = True,",
        "    has_py2_only_sources = True,",
        ")",
        "py_library(",
        "    name = 'pylib',",
        "    srcs = ['pylib.py'],",
        "    deps = [':loweruserlib'],",
        "    srcs_version = 'PY3ONLY'",
        ")",
        "userlib(",
        "    name = 'upperuserlib',",
        "    srcs = ['upperuserlib.py'],",
        "    deps = [':pylib'],",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//pkg:upperuserlib");
    StructImpl legacyInfo = PyProviderUtils.getLegacyProvider(target);
    PyInfo modernInfo = PyProviderUtils.getModernProvider(target);

    assertThat(PyStructUtils.getTransitiveSources(legacyInfo))
        .containsExactly(
            getSourceArtifact("pkg/loweruserlib.py"),
            getSourceArtifact("pkg/pylib.py"),
            getSourceArtifact("pkg/upperuserlib.py"));
    assertThat(PyStructUtils.getUsesSharedLibraries(legacyInfo)).isTrue();
    assertThat(PyStructUtils.getHasPy2OnlySources(legacyInfo)).isTrue();
    assertThat(PyStructUtils.getHasPy3OnlySources(legacyInfo)).isTrue();

    assertThat(modernInfo.getTransitiveSources().getSet(Artifact.class))
        .containsExactly(
            getSourceArtifact("pkg/loweruserlib.py"),
            getSourceArtifact("pkg/pylib.py"),
            getSourceArtifact("pkg/upperuserlib.py"));
    assertThat(modernInfo.getUsesSharedLibraries()).isTrue();
    assertThat(modernInfo.getHasPy2OnlySources()).isTrue();
    assertThat(modernInfo.getHasPy3OnlySources()).isTrue();
  }
}
