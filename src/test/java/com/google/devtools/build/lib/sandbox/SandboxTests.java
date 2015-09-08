// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import com.google.common.base.Predicates;
import com.google.devtools.build.lib.testutil.BlazeTestSuiteBuilder;
import com.google.devtools.build.lib.testutil.CustomSuite;

import org.junit.runner.RunWith;

import java.util.Set;

/**
 * Test suite that runs all tests that are not local-only.
 */
@RunWith(CustomSuite.class)
public class SandboxTests extends BlazeTestSuiteBuilder {
  public static Set<Class<?>> suite() {
    return new SandboxTests()
        .getBuilder()
        .matchClasses(Predicates.not(BlazeTestSuiteBuilder.TEST_IS_LOCAL_ONLY))
        .create();
  }
}
