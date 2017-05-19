// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkFileset and SkylarkFileType.
 */
@RunWith(JUnit4.class)
public class SkylarkFileHelperTest extends SkylarkTestCase {

  @Before
  public final void createBuildFile() throws Exception  {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        "  cmd = 'dummy_cmd',",
        "  srcs = ['a.txt', 'b.img'],",
        "  tools = ['t.exe'],",
        "  outs = ['c.txt'])");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testFilterPasses() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "FileType(['.img']).filter(ruleContext.files.srcs)");
    assertEquals("b.img", ActionsTestUtil.baseNamesOf((Iterable<Artifact>) result));
  }

  @Test
  public void testFilterFiltersFilesOut() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    Object result =
        evalRuleContextCode(ruleContext, "FileType(['.xyz']).filter(ruleContext.files.srcs)");
    assertThat(((Iterable<?>) result)).isEmpty();
  }

  @Test
  public void testArtifactPath() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    String result = (String) evalRuleContextCode(ruleContext, "ruleContext.files.tools[0].path");
    assertEquals("foo/t.exe", result);
  }

  @Test
  public void testArtifactShortPath() throws Exception {
    SkylarkRuleContext ruleContext = createRuleContext("//foo:foo");
    String result =
        (String) evalRuleContextCode(ruleContext, "ruleContext.files.tools[0].short_path");
    assertEquals("foo/t.exe", result);
  }
}
