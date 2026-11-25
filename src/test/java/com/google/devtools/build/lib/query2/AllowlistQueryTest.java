// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.allowlisting.AllowlistDummyRule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.util.MockToolsConfig;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.testutil.AbstractQueryTest.QueryHelper;
import com.google.devtools.build.lib.query2.testutil.SkyframeQueryHelper;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests to verify allow list dependencies are reported in blaze query. */
@RunWith(JUnit4.class)
public class AllowlistQueryTest {
  @Test
  public void testAllowlistDependenciesAreReported() throws Exception {
    QueryHelper<Target> helper =
        new SkyframeQueryHelper() {
          @Override
          protected String getRootDirectoryNameForSetup() {
            return "/workspace";
          }

          @Override
          protected void performAdditionalClientSetup(MockToolsConfig mockToolsConfig)
              throws IOException {}

          @Override
          protected ImmutableList<QueryFunction> getExtraQueryFunctions() {
            return ImmutableList.of();
          }
        };
    helper.setUp();
    helper.setUniverseScope("//...:*");
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(AllowlistDummyRule.DEFINITION);
    helper.useRuleClassProvider(builder.build());
    helper.writeFile(
        "suballowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//x",
            ],
        )
        """);
    helper.writeFile(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            includes = [
                "//suballowlist:allowlist",
            ],
            packages = [
                "//x",
            ],
        )
        """);
    helper.writeFile("x/BUILD", "rule_with_allowlist(name = 'x')");
    Set<Target> results = helper.evaluateQuery("deps(//x:x)").getResultSet();
    assertThat(results.stream().map(r -> helper.getLabel(r)).collect(toImmutableList()))
        .containsExactly("//x:x", "//allowlist:allowlist", "//suballowlist:allowlist");
  }
}
