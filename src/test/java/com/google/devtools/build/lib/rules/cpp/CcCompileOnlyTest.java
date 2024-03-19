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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.CompileOnlyTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests that validate --compile_only behavior.
 */
@RunWith(JUnit4.class)
public class CcCompileOnlyTest extends CompileOnlyTestCase {

  @Test
  public void testCcCompileOnly() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig, CcToolchainConfig.builder().withFeatures(CppRuleClasses.SUPPORTS_PIC));
    useConfiguration("--platforms=" + TestConstants.PLATFORM_LABEL);
    scratch.file("package/BUILD",
        "cc_binary(name='foo', srcs=['foo.cc', ':bar'], deps = [':foolib'])",
        "cc_library(name='foolib', srcs=['foolib.cc'])",
        "genrule(name='bar', outs=['bar.h', 'bar.cc'], cmd='touch $(OUTS)')");
    scratch.file("package/foo.cc",
        "#include <stdio.h>",
        "int main() {",
        "  printf(\"Hello, world!\\n\");",
        "  return 0;",
        "}");
    scratch.file("package/foolib.cc",
        "#include <stdio.h>",
        "int printHeader() {",
        "  printf(\"Hello, library!\\n\");",
        "  return 0;",
        "}");

    ConfiguredTarget target = getConfiguredTarget("//package:foo");

    assertThat(getArtifactByExecPathSuffix(target, "/foo.pic.o")).isNotNull();
    assertThat(getArtifactByExecPathSuffix(target, "/bar.pic.o")).isNotNull();
    // Check that deps are not built
    assertThat(getArtifactByExecPathSuffix(target, "/foolib.pic.o")).isNull();
    // Check that linking is not executed
    assertThat(getArtifactByExecPathSuffix(target, "/foo")).isNull();
  }
}
