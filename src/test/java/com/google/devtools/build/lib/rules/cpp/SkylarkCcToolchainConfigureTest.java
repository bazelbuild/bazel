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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc autoconfiguration. */
@RunWith(JUnit4.class)
public class SkylarkCcToolchainConfigureTest extends EvaluationTestCase {

  @Test
  public void testActionNames() throws Exception {
    newTest()
        .testStatement("COMPILE_ACTIONS", MutableList.copyOf(env, CcCommon.ALL_COMPILE_ACTIONS))
        .testStatement("LINK_ACTIONS", MutableList.copyOf(env, CcCommon.ALL_LINK_ACTIONS))
        .testStatement("ARCHIVE_ACTIONS", MutableList.copyOf(env, CcCommon.ALL_ARCHIVE_ACTIONS))
        .testStatement("OTHER_ACTIONS", MutableList.copyOf(env, CcCommon.ALL_OTHER_ACTIONS));
  }

  @Test
  public void testSplitEscaped() throws Exception {
    newTest()
        .testStatement("split_escaped('a:b:c', ':')", MutableList.of(env, "a", "b", "c"))
        .testStatement("split_escaped('a%:b', ':')", MutableList.of(env, "a:b"))
        .testStatement("split_escaped('a%%b', ':')", MutableList.of(env, "a%b"))
        .testStatement("split_escaped('a:::b', ':')", MutableList.of(env, "a", "", "", "b"))
        .testStatement("split_escaped('a:b%:c', ':')", MutableList.of(env, "a", "b:c"))
        .testStatement("split_escaped('a%%:b:c', ':')", MutableList.of(env, "a%", "b", "c"))
        .testStatement("split_escaped(':a', ':')", MutableList.of(env, "", "a"))
        .testStatement("split_escaped('a:', ':')", MutableList.of(env, "a", ""))
        .testStatement("split_escaped('::a::', ':')", MutableList.of(env, "", "", "a", "", ""))
        .testStatement("split_escaped('%%%:a%%%%:b', ':')", MutableList.of(env, "%:a%%", "b"))
        .testStatement("split_escaped('', ':')", MutableList.of(env, ""))
        .testStatement("split_escaped('%', ':')", MutableList.of(env, "%"))
        .testStatement("split_escaped('%%', ':')", MutableList.of(env, "%"))
        .testStatement("split_escaped('%:', ':')", MutableList.of(env, ":"))
        .testStatement("split_escaped(':', ':')", MutableList.of(env, "", ""))
        .testStatement("split_escaped('a%%b', ':')", MutableList.of(env, "a%b"))
        .testStatement("split_escaped('a%:', ':')", MutableList.of(env, "a:"));
  }

  @Test
  public void testActionConfig() throws Exception {
    newTest()
        .testStatement(
            "action_config('c++-compile', '/usr/bin/gcc')",
            "\n"
                + "  action_config {\n"
                + "    config_name: 'c++-compile'\n"
                + "    action_name: 'c++-compile'\n"
                + "    tool {\n"
                + "      tool_path: '/usr/bin/gcc'\n"
                + "    }\n"
                + "  }");
  }

  @Test
  public void testFeature() throws Exception {
    newTest()
        .testStatement(
            "feature("
                + "'fully_static_link', "
                + " [ "
                + "    flag_set("
                + "      ['c++-link-dynamic-library', 'c++-link-nodeps-dynamic-library'], "
                + "      [flag_group([flag('-a'), flag('-b'), flag('-c')])])])",
            "\n"
                + "  feature {\n"
                + "    name: 'fully_static_link'\n"
                + "    enabled: true\n"
                + "    flag_set {\n"
                + "      action: 'c++-link-dynamic-library'\n"
                + "      action: 'c++-link-nodeps-dynamic-library'\n"
                + "      flag_group {\n"
                + "        flag: '-a'\n"
                + "        flag: '-b'\n"
                + "        flag: '-c'\n"
                + "      }\n"
                + "    }\n"
                + "  }");
  }

  @Test
  public void testFeatureThoroughly() throws Exception {
    newTest()
        .testStatement(
            "feature("
                + "'fully_static_link', "
                + " [ "
                + "   flag_set("
                + "     ['c++-link-dynamic-library'], "
                + "     [flag_group([flag('-a')])]),"
                + "   flag_set("
                + "     ['c++-link-dynamic-library'],"
                + "     ["
                + "        flag_group("
                + "          [flag('-a')],"
                + "          iterate_over='a'),"
                + "        flag_group("
                + "          [flag('-c')],"
                + "          expand_if_all_available=['a','b'],"
                + "          expand_if_none_available=['a'],"
                + "          expand_if_true=['a','b'],"
                + "          expand_if_false=['a'],"
                + "          expand_if_equal=[['a','val']],"
                + "        ),"
                + "        flag_group("
                + "          [flag('-c')],"
                + "          iterate_over='a',"
                + "          expand_if_all_available=['a','b'],"
                + "          expand_if_none_available=['a'],"
                + "          expand_if_true=['a','b'],"
                + "          expand_if_false=['a'],"
                + "          expand_if_equal=[['a','val']],"
                + "        )"
                + "      ]),"
                + "    flag_set("
                + "      ['c++-link-dynamic-library'], "
                + "      [flag_group([flag_group([flag('-a')])])])"
                + " ])",
            "\n"
                + "  feature {\n"
                + "    name: 'fully_static_link'\n"
                + "    enabled: true\n"
                + "    flag_set {\n"
                + "      action: 'c++-link-dynamic-library'\n"
                + "      flag_group {\n"
                + "        flag: '-a'\n"
                + "      }\n"
                + "    }\n"
                + "    flag_set {\n"
                + "      action: 'c++-link-dynamic-library'\n"
                + "      flag_group {\n"
                + "        iterate_over: 'a'\n"
                + "        flag: '-a'\n"
                + "      }\n"
                + "      flag_group {\n"
                + "        expand_if_all_available: 'a'\n"
                + "        expand_if_all_available: 'b'\n"
                + "        expand_if_none_available: 'a'\n"
                + "        expand_if_true: 'a'\n"
                + "        expand_if_true: 'b'\n"
                + "        expand_if_false: 'a'\n"
                + "        expand_if_equal { variable: 'a' value: 'val' }\n"
                + "        flag: '-c'\n"
                + "      }\n"
                + "      flag_group {\n"
                + "        expand_if_all_available: 'a'\n"
                + "        expand_if_all_available: 'b'\n"
                + "        expand_if_none_available: 'a'\n"
                + "        expand_if_true: 'a'\n"
                + "        expand_if_true: 'b'\n"
                + "        expand_if_false: 'a'\n"
                + "        expand_if_equal { variable: 'a' value: 'val' }\n"
                + "        iterate_over: 'a'\n"
                + "        flag: '-c'\n"
                + "      }\n"
                + "    }\n"
                + "    flag_set {\n"
                + "      action: 'c++-link-dynamic-library'\n"
                + "      flag_group {\n"
                + "      flag_group {\n"
                + "        flag: '-a'\n"
                + "      }\n"
                + "      }\n"
                + "    }\n"
                + "  }");
  }

  private ModalTestCase newTest(String... skylarkOptions) throws IOException {
    return new SkylarkTest(skylarkOptions)
        // A mock implementation of Label to be able to parse lib_cc_configure under default
        // Skylark environment (lib_cc_configure is meant to be used from the repository
        // environment).
        .setUp("def Label(arg):\n  return 42")
        .setUp(
            ResourceLoader.readFromResources(
                TestConstants.BAZEL_REPO_PATH + "tools/cpp/lib_cc_configure.bzl"))
        .setUp(
            ResourceLoader.readFromResources(
                TestConstants.BAZEL_REPO_PATH + "tools/cpp/crosstool_utils.bzl"));
  }
}
