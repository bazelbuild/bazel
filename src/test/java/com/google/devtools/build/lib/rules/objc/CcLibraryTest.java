// Copyright 2021 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc_library with Apple-specific logic. */
@RunWith(JUnit4.class)
public class CcLibraryTest extends ObjcRuleTestCase {
  @Override
  protected ScratchAttributeWriter createLibraryTargetWriter(String labelString) {
    return ScratchAttributeWriter.fromLabelString(this, "cc_library", labelString);
  }

  @Test
  public void testGenerateDsymFlagPropagatesToCcLibraryFeature() throws Exception {
    useConfiguration("--apple_generate_dsym");
    createLibraryTargetWriter("//cc/lib").setList("srcs", "a.cc").write();
    CommandAction compileAction = compileAction("//cc/lib", "a.o");
    assertThat(compileAction.getArguments()).contains("-DDUMMY_GENERATE_DSYM_FILE");
  }
}
