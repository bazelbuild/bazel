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

import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Legacy test: These tests test --experimental_objc_crosstool=off. See README.
 */
@RunWith(JUnit4.class)
public class LegacyIosExtensionBinaryTest extends IosExtensionBinaryTest {
  @Override
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.OFF;
  }

  @Test
  public void testLinkActionWithTransitiveCppDependency() throws Exception {
    checkLinkActionWithTransitiveCppDependency(RULE_TYPE, EXTRA_LINK_ARGS);
  }

  @Test
  public void testCompilesSourcesWithModuleMapsEnabled() throws Exception {
    checkCompilesSourcesWithModuleMapsEnabled(RULE_TYPE);
  }

  @Test
  public void testLinkWithFrameworkImportsIncludesFlagsAndInputArtifacts() throws Exception {
    checkLinkWithFrameworkImportsIncludesFlagsAndInputArtifacts(RULE_TYPE);
  }

  @Test
  public void testForceLoadsAlwayslinkTargets() throws Exception {
    checkForceLoadsAlwayslinkTargets(RULE_TYPE, EXTRA_LINK_ARGS);
  }

  @Test
  public void testReceivesTransitivelyPropagatedDefines() throws Exception {
    checkReceivesTransitivelyPropagatedDefines(RULE_TYPE);
  }

  @Test
  public void testSdkIncludesUsedInCompileAction() throws Exception {
    checkSdkIncludesUsedInCompileAction(RULE_TYPE);
  }
}
