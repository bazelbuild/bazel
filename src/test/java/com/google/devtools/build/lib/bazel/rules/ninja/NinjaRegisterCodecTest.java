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

package com.google.devtools.build.lib.bazel.rules.ninja;

import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScopeRegister;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests serialization of {@link NinjaScopeRegister}. */
@RunWith(JUnit4.class)
public class NinjaRegisterCodecTest {
  @Test
  public void testCodec() throws Exception {
    NinjaScopeRegister register = NinjaScopeRegister.create();
    NinjaScope mainScope = register.getMainScope();
    NinjaScope childScope = mainScope.addIncluded(register, 11);
    NinjaScope subNinjaScope = childScope.addSubNinja(register, 12);

    mainScope.addExpandedVariable(1, "main_var", "aaa");
    childScope.addExpandedVariable(2, "child_var", "bbb");
    subNinjaScope.addExpandedVariable(3, "sub_var", "ccc");

    NinjaScopeRegister frozen = register.freeze();

    new SerializationTester(frozen).runTests();
  }
}
