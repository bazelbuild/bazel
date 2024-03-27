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

package com.google.devtools.build.lib.analysis.allowlisting;

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.util.AnalysisCachingTestBase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that allowlists are invalidated after change. */
@RunWith(JUnit4.class)
public final class AllowlistCachingTest extends AnalysisCachingTestBase {

  @Before
  public void addDummyRule() throws Exception {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(AllowlistDummyRule.DEFINITION);
    useRuleClassProvider(builder.build());
  }

  @Test
  public void testStillCorrectAfterChangesToAllowlist() throws Exception {
    scratch.file("allowlist/BUILD", "package_group(name='allowlist', packages=[])");
    scratch.file("x/BUILD", "rule_with_allowlist(name='x')");

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//x:x"));
    assertContainsEvent("Dummy is not available.");
    eventCollector.clear();
    reporter.addHandler(failFastHandler);
    scratch.overwriteFile(
        "allowlist/BUILD",
        """
        package_group(
            name = "allowlist",
            packages = [
                "//...",
            ],
        )
        """);
    update("//x:x");
    assertNoEvents();
  }
}
