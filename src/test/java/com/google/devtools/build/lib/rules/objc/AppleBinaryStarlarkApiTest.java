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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import java.util.Set;
import org.junit.Before;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test cases for the Starlark Apple Linking API, {@code apple_common.link_multi_arch_binary}. These
 * tests verify that the API has parity with the native {@code apple_binary} rule.
 *
 * <p>This is a subclass of {@link AppleBinaryTest}, to ensure it runs all tests of that suite --
 * the main difference is that {@link AppleBinaryTest} tests the native rule, and this suite tests
 * using a Starlark stub around the Starlark API.
 */
@RunWith(JUnit4.class)
public class AppleBinaryStarlarkApiTest extends AppleBinaryTest {

  static final RuleType RULE_TYPE =
      new RuleType("apple_binary_starlark") {
        @Override
        Iterable<String> requiredAttributes(
            Scratch scratch, String packageDir, Set<String> alreadyAdded) throws IOException {
          return Iterables.concat(
              ImmutableList.of(),
              AppleBinaryTest.RULE_TYPE.requiredAttributes(scratch, packageDir, alreadyAdded));
        }

        @Override
        public String starlarkLoadPrerequisites() {
          return "load('//test_starlark:apple_binary_starlark.bzl', 'apple_binary_starlark')";
        }
      };

  @Before
  public final void setup() throws Exception  {
    scratch.file("test_starlark/BUILD");
    String toolsRepo = TestConstants.TOOLS_REPOSITORY;
    String toolsLoc = toolsRepo + "//tools/objc";

    scratch.file(
        "test_starlark/apple_binary_starlark.bzl",
        "def apple_binary_starlark_impl(ctx):",
        "  binary_output = apple_common.link_multi_arch_binary(ctx=ctx)",
        "  return [",
        "      DefaultInfo(",
        "          files=depset([binary_output.binary_provider.binary])),",
        "      OutputGroupInfo(**binary_output.output_groups),",
        "      binary_output.binary_provider,",
        "      binary_output.debug_outputs_provider",
        "  ]",
        "apple_binary_starlark = rule(",
        "    apple_binary_starlark_impl,",
        "    attrs = {",
        "        '_child_configuration_dummy': attr.label(",
        "            cfg=apple_common.multi_arch_split,",
        "            default=Label('" + toolsRepo + "//tools/cpp:current_cc_toolchain'),),",
        "        '_cc_toolchain': attr.label(",
        "            default=Label('" + toolsRepo + "//tools/cpp:current_cc_toolchain'),),",
        "        '_googlemac_proto_compiler': attr.label(",
        "            cfg='host',",
        "            default=Label('" + toolsLoc + ":protobuf_compiler_wrapper')),",
        "        '_googlemac_proto_compiler_support': attr.label(",
        "            cfg='host',",
        "            default=Label('" + toolsLoc + ":protobuf_compiler_support')),",
        "        '_lib_protobuf': attr.label(",
        "            default=Label('" + toolsLoc + ":protobuf_lib')),",
        "        '_protobuf_well_known_types': attr.label(",
        "            cfg='host',",
        "            default=Label('" + toolsLoc + ":protobuf_well_known_types')),",
        "        '_xcode_config': attr.label(",
        "            default=configuration_field(",
        "                fragment='apple', name='xcode_config_label'),),",
        "        '_xcrunwrapper': attr.label(",
        "            executable=True,",
        "            cfg='host',",
        "            default=Label('" + toolsLoc + ":xcrunwrapper')),",
        "        'binary_type': attr.string(),",
        "        'bundle_loader': attr.label(aspects=[apple_common.objc_proto_aspect],),",
        "        'deps': attr.label_list(",
        "             cfg=apple_common.multi_arch_split,",
        "             aspects=[apple_common.objc_proto_aspect],",
        "        ),",
        "        'dylibs': attr.label_list(aspects=[apple_common.objc_proto_aspect],),",
        "        'linkopts': attr.string_list(),",
        "        'platform_type': attr.string(),",
        "        'minimum_os_version': attr.string(),",
        "        'should_lipo': attr.bool(),",
        "        'stamp': attr.int(values=[-1,0,1],default=-1),",
        "    },",
        "    fragments = ['apple', 'objc', 'cpp',],",
        ")");
  }

  @Override
  protected RuleType getRuleType() {
    return RULE_TYPE;
  }

  @Override
  public void testMinimumOs_invalid_containsAlphabetic() throws Exception {
    // TODO(b/70937317): Disabled due to different error handling for Starlark rule.
  }

  @Override
  public void testMinimumOs_invalid_tooManyComponents() throws Exception {
    // TODO(b/70937317): Disabled due to different error handling for Starlark rule.
  }

  @Override
  public void testMinimumOs_invalid_nonVersion() throws Exception {
    // TODO(b/70937317): Disabled due to different error handling for Starlark rule.
  }

  @Override
  public void testBundleLoaderCantBeSetWithoutBundleBinaryType() throws Exception {
    // TODO(b/70937317): Disabled due to different error handling for Starlark rule.
  }
}
