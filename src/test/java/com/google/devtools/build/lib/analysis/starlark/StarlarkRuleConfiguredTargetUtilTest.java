// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;

import static com.google.common.truth.Truth.assertThat;

public class StarlarkRuleConfiguredTargetUtilTest extends BuildViewTestCase {

    @Test
    public void testForwardingDefaultInfoRetainsDataRunfiles() throws Exception {
        scratch.file("foo/rules.bzl",
                "def _forward_default_info_impl(ctx):",
                "    return [",
                "        ctx.attr.target[DefaultInfo],",
                "    ]",
                "forward_default_info = rule(",
                "    implementation = _forward_default_info_impl,",
                "    attrs = {",
                "        'target': attr.label(",
                "            mandatory = True,",
                "        ),",
                "    },",
                ")"
        );
        scratch.file("foo/i_am_a_runfile");
        scratch.file("foo/BUILD",
                "load(':rules.bzl', 'forward_default_info')",
                "java_library(",
                "    name = 'lib',",
                "    data = ['i_am_a_runfile'],",
                ")",
                "forward_default_info(",
                "    name = 'forwarded_lib',",
                "    target = ':lib',",
                ")"
        );
        ConfiguredTarget nativeTarget = getConfiguredTarget("//foo:lib");
        ImmutableList<Artifact> nativeRunfiles = getDataRunfiles(nativeTarget).getAllArtifacts().toList();
        ConfiguredTarget forwardedTarget = getConfiguredTarget("//foo:forwarded_lib");
        ImmutableList<Artifact> forwardedRunfiles = getDataRunfiles(forwardedTarget).getAllArtifacts().toList();
        assertThat(forwardedRunfiles).isEqualTo(nativeRunfiles);
        assertThat(forwardedRunfiles).hasSize(1);
        assertThat(forwardedRunfiles.get(0).getPath().getBaseName()).isEqualTo("i_am_a_runfile");
    }
}
