// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skydoc.rendering.RuleInfo;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Java tests for Skydoc.
 */
@RunWith(JUnit4.class)
public final class SkydocTest extends SkylarkTestCase {

  @Test
  public void testRuleInfoAttrs() throws Exception {
    Path file =
        scratch.file(
            "/test/test.bzl",
            "def rule_impl(ctx):",
            "  return struct()",
            "",
            "my_rule = rule(",
            "    doc = 'This is my rule. It does stuff.',",
            "    implementation = rule_impl,",
            "    attrs = {",
            "        'first': attr.label(mandatory=True, allow_files=True, single_file=True),",
            "        'second': attr.string_dict(mandatory=True),",
            "        'third': attr.output(mandatory=True),",
            "        'fourth': attr.bool(default=False, mandatory=False),",
            "    },",
            ")");
    byte[] bytes = FileSystemUtils.readWithKnownFileSize(file, file.getFileSize());

    ParserInputSource parserInputSource =
        ParserInputSource.create(bytes, file.asFragment());

    List<RuleInfo> ruleInfos = new SkydocMain().eval(parserInputSource);
    assertThat(ruleInfos).hasSize(1);

    RuleInfo ruleInfo = Iterables.getOnlyElement(ruleInfos);
    assertThat(ruleInfo.getDocString()).isEqualTo("This is my rule. It does stuff.");
    assertThat(ruleInfo.getAttrNames()).containsExactly(
        "first", "second", "third", "fourth");
  }
}
