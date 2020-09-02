// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.syntax.EvalException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test class for {@link AttributeValueSource}.
 */
@RunWith(JUnit4.class)
public class AttributeValueSourceTest {

  @Test
  public void testValidateStarlarkName() throws Exception {
    // Success means "no exception is being thrown".
    AttributeValueSource.COMPUTED_DEFAULT.validateStarlarkName("_name");
    AttributeValueSource.LATE_BOUND.validateStarlarkName("_name");
    AttributeValueSource.DIRECT.validateStarlarkName("_name");
    AttributeValueSource.DIRECT.validateStarlarkName("name");
  }

  @Test
  public void testValidateStarlarkName_emptyName() throws Exception {
    for (AttributeValueSource source : AttributeValueSource.values()) {
      assertNameIsNotValid(source, "", "Attribute name must not be empty.");
    }
  }

  @Test
  public void testValidateStarlarkName_missingPrefix() throws Exception {
    String msg =
        "When an attribute value is a function, the attribute must be private "
            + "(i.e. start with '_'). Found 'my_name'";
    assertNameIsNotValid(AttributeValueSource.COMPUTED_DEFAULT, "my_name", msg);
    assertNameIsNotValid(AttributeValueSource.LATE_BOUND, "my_name", msg);
  }

  private void assertNameIsNotValid(
      AttributeValueSource source, String name, String expectedExceptionMessage) throws Exception {
    EvalException ex = assertThrows(EvalException.class, () -> source.validateStarlarkName(name));
    assertThat(ex).hasMessageThat().isEqualTo(expectedExceptionMessage);
  }

  @Test
  public void testConvertToNativeName() throws Exception {
    assertConvertsToCorrectNativeName(AttributeValueSource.COMPUTED_DEFAULT, "_name", "$name");
    assertConvertsToCorrectNativeName(AttributeValueSource.LATE_BOUND, "_name", ":name");
    assertConvertsToCorrectNativeName(AttributeValueSource.DIRECT, "_name", "$name");
    assertConvertsToCorrectNativeName(AttributeValueSource.DIRECT, "name", "name");
  }

  private void assertConvertsToCorrectNativeName(
      AttributeValueSource source, String starlarkName, String expectedNativeName)
      throws Exception {
    assertThat(source.convertToNativeName(starlarkName)).isEqualTo(expectedNativeName);
  }

  @Test
  public void testConvertToNativeName_invalidName() throws Exception {
    assertTranslationFails(AttributeValueSource.COMPUTED_DEFAULT, "name");
    assertTranslationFails(AttributeValueSource.LATE_BOUND, "name");
  }

  private void assertTranslationFails(AttributeValueSource source, String invalidName)
      throws Exception {
    EvalException ex =
        assertThrows(EvalException.class, () -> source.convertToNativeName(invalidName));
    assertThat(ex)
        .hasMessageThat()
        .isEqualTo(
            String.format(
                "When an attribute value is a function, the attribute must be private "
                    + "(i.e. start with '_'). Found '%s'",
                invalidName));
  }

  @Test
  public void testBuilderGetValueSource() throws Exception {
    assertBuilderHasCorrectSource(COMPUTED_DEFAULT_BUILDER, AttributeValueSource.COMPUTED_DEFAULT);
    assertBuilderHasCorrectSource(LATE_BOUND_BUILDER, AttributeValueSource.LATE_BOUND);
    assertBuilderHasCorrectSource(DIRECT_BUILDER, AttributeValueSource.DIRECT);
  }

  private void assertBuilderHasCorrectSource(
      Attribute.Builder<?> builder, AttributeValueSource expectedSource) throws Exception {
    assertThat(builder.getValueSource()).isEqualTo(expectedSource);
  }

  private static final Attribute.Builder<?> COMPUTED_DEFAULT_BUILDER =
      attr("x", STRING)
          .value(
              new ComputedDefault() {
                @Override
                public Object getDefault(AttributeMap rule) {
                  return null;
                }
              });

  private static final Attribute.Builder<?> LATE_BOUND_BUILDER =
      attr(":x", STRING).value(LateBoundDefault.alwaysNull());

  private static final Attribute.Builder<?> DIRECT_BUILDER = attr("x", STRING).value("value");
}
