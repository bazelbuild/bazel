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
package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MethodOptionDefinition}. */
@RunWith(JUnit4.class)
public class MethodOptionDefinitionTest {
  /** Dummy options class for testing method-based options. */
  @OptionsClass
  public abstract static class MethodOptionsTestFields extends OptionsBase {
    @Option(
        name = "foo",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "42")
    public abstract int getFoo();
  }

  @Test
  public void testMethodOptionParsing() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(MethodOptionsTest.class).build();

    parser.parse("--foo=123");

    MethodOptionsTest options = parser.getOptions(MethodOptionsTest.class);
    assertThat(options.getFoo()).isEqualTo(123);
  }

  @Test
  public void testGeneratedClassGettersAndSetters() {
    MethodOptionsTest options = new MethodOptionsTest();
    options.setFoo(123);
    assertThat(options.getFoo()).isEqualTo(123);
  }

  @Test
  public void testMethodOptionDefinitionAccess() throws Exception {
    MethodOptionsTest options = new MethodOptionsTest();
    OptionDefinition fooDefinition = MethodOptionDefinition.get(MethodOptionsTest.class, "getFoo");

    fooDefinition.setValue(options, 456);
    assertThat(options.getFoo()).isEqualTo(456);
    assertThat(fooDefinition.getRawValue(options)).isEqualTo(456);
  }

  @Test
  public void getDeclaringClass_returnsImplementation() throws Exception {
    MethodOptionDefinition definition =
        MethodOptionDefinition.get(MethodOptionsTest.class, "getFoo");

    // The important part is that the return value of getDeclaringClass() can be passed to
    // getOptions(), but it's nice to not have this test depend on OptionsParser.
    assertThat(definition.getDeclaringClass(OptionsBase.class)).isEqualTo(MethodOptionsTest.class);
  }
}
