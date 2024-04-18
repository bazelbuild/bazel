// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link OutputFileConfiguredTarget}. */
@RunWith(JUnit4.class)
public class OutputFileConfiguredTargetTest extends BuildViewTestBase {
  @Test
  public void generatingRuleIsCorrect() throws Exception {
    scratch.file(
        "foo/BUILD",
        """
        genrule(
            name = "generating_rule",
            srcs = [],
            outs = ["generated.source"],
            cmd = "echo hi > $@",
        )
        """);
    update("//foo:generating_rule");
    OutputFileConfiguredTarget generatedSource =
        (OutputFileConfiguredTarget)
            getConfiguredTarget("//foo:generated.source", getExecConfiguration());
    assertThat(generatedSource.getGeneratingRule())
        .isSameInstanceAs(getConfiguredTarget("//foo:generating_rule", getExecConfiguration()));
  }
}
