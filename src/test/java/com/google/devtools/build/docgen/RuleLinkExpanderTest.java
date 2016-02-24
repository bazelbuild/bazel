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
package com.google.devtools.build.docgen;

import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableMap;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link RuleLinkExpander}. */
@RunWith(JUnit4.class)
public class RuleLinkExpanderTest {
  private RuleLinkExpander expander;

  @Before public void setUp() {
    expander = new RuleLinkExpander(ImmutableMap.<String, String>builder()
        .put("cc_library", "c-cpp")
        .put("cc_binary", "c-cpp")
        .put("java_binary", "java")
        .put("proto_library", "protocol-buffer")
        .build());
  }

  @Test public void testRule() {
    String docs = "<a href=\"${link java_binary}\">java_binary rule</a>";
    String expected = "<a href=\"java.html#java_binary\">java_binary rule</a>";
    assertEquals(expected, expander.expand(docs));
  }

  @Test public void testRuleAndAttribute() {
    String docs = "<a href=\"${link java_binary.runtime_deps}\">runtime_deps attribute</a>";
    String expected = "<a href=\"java.html#java_binary.runtime_deps\">runtime_deps attribute</a>";
    assertEquals(expected, expander.expand(docs));
  }

  @Test public void testRuleExamples() {
    String docs = "<a href=\"${link cc_binary_examples}\">examples</a>";
    String expected = "<a href=\"c-cpp.html#cc_binary_examples\">examples</a>";
    assertEquals(expected, expander.expand(docs));
  }

  @Test public void testRuleArgs() {
    String docs = "<a href=\"${link cc_binary_args}\">args</a>";
    String expected = "<a href=\"c-cpp.html#cc_binary_args\">args</a>";
    assertEquals(expected, expander.expand(docs));
  }

  @Test public void testStaticPageRef() {
    String docs = "<a href=\"${link common-definitions}\">Common Definitions</a>";
    String expected = "<a href=\"common-definitions.html\">Common Definitions</a>";
    assertEquals(expected, expander.expand(docs));
  }

  @Test public void testStaticPageWithHeadingRef() {
    String docs = "<a href=\"${link common-definitions.label-expansion}\">Label Expansion</a>";
    String expected = "<a href=\"common-definitions.html#label-expansion\">Label Expansion</a>";
    assertEquals(expected, expander.expand(docs));
  }

  @Test(expected = IllegalArgumentException.class)
  public void testRefNotFound() {
    String docs = "<a href=\"${link foo.bar}\">bar</a>";
    expander.expand(docs);
  }
}
