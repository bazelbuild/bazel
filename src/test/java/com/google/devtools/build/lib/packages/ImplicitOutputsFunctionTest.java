// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.AttributeValueGetter;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ImplicitOutputsFunction}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public final class ImplicitOutputsFunctionTest {
  private void assertPlaceholderCollection(
      String template, String expectedTemplate, String... expectedPlaceholders) throws Exception {
    List<String> actualPlaceholders = new ArrayList<>();
    assertThat(
            ImplicitOutputsFunction.createPlaceholderSubstitutionFormatString(
                template, actualPlaceholders))
        .isEqualTo(expectedTemplate);
    assertThat(actualPlaceholders)
        .containsExactlyElementsIn(Arrays.asList(expectedPlaceholders))
        .inOrder();
  }

  @Test
  public void testNoPlaceholder() throws Exception {
    assertPlaceholderCollection("foo", "foo");
  }

  @Test
  public void testJustPlaceholder() throws Exception {
    assertPlaceholderCollection("%{foo}", "%s", "foo");
  }

  @Test
  public void testPrefixedPlaceholder() throws Exception {
    assertPlaceholderCollection("foo%{bar}", "foo%s", "bar");
  }

  @Test
  public void testSuffixedPlaceholder() throws Exception {
    assertPlaceholderCollection("%{foo}bar", "%sbar", "foo");
  }

  @Test
  public void testMultiplePlaceholdersPrefixed() throws Exception {
    assertPlaceholderCollection("foo%{bar}baz%{qux}", "foo%sbaz%s", "bar", "qux");
  }

  @Test
  public void testMultiplePlaceholdersSuffixed() throws Exception {
    assertPlaceholderCollection("%{foo}bar%{baz}qux", "%sbar%squx", "foo", "baz");
  }

  @Test
  public void testTightlyPackedPlaceholders() throws Exception {
    assertPlaceholderCollection("%{foo}%{bar}%{baz}", "%s%s%s", "foo", "bar", "baz");
  }

  @Test
  public void testIncompletePlaceholder() throws Exception {
    assertPlaceholderCollection("%{foo", "%%{foo");
  }

  @Test
  public void testCompleteAndIncompletePlaceholder() throws Exception {
    assertPlaceholderCollection("%{foo}%{bar", "%s%%{bar", "foo");
  }

  @Test
  public void testPlaceholderLooksLikeNestedIncompletePlaceholder() throws Exception {
    assertPlaceholderCollection("%{%{foo", "%%{%%{foo");
  }

  @Test
  public void testPlaceholderLooksLikeNestedPlaceholder() throws Exception {
    assertPlaceholderCollection("%{%{foo}", "%s", "%{foo");
  }

  @Test
  public void testEscapesJustPercentSign() throws Exception {
    assertPlaceholderCollection("%", "%%");
  }

  @Test
  public void testEscapesPrintfPlaceholder() throws Exception {
    assertPlaceholderCollection("%{x}%s%{y}", "%s%%s%s", "x", "y");
  }

  @Test
  public void testEscapesPercentSign() throws Exception {
    assertPlaceholderCollection("foo%{bar}%baz", "foo%s%%baz", "bar");
  }

  private static AttributeValueGetter attrs(
      final Map<String, ? extends Collection<String>> values) {
    return new AttributeValueGetter() {
      @Override
      public Set<String> get(AttributeMap ignored, String attr) {
        return new LinkedHashSet<>(Preconditions.checkNotNull(values.get(attr)));
      }
    };
  }

  private void assertPlaceholderSubtitution(
      String template,
      AttributeValueGetter attrValues,
      String[] expectedSubstitutions,
      String[] expectedFoundPlaceholders)
      throws Exception {
    // Directly call into ParsedTemplate in order to access the attribute names.
    ImplicitOutputsFunction.ParsedTemplate parsedTemplate =
        ImplicitOutputsFunction.ParsedTemplate.parse(template);

    assertThat(parsedTemplate.attributeNames())
        .containsExactlyElementsIn(Arrays.asList(expectedFoundPlaceholders))
        .inOrder();

    // Test the actual substitution code.
    List<String> substitutions =
        ImplicitOutputsFunction.substitutePlaceholderIntoTemplate(template, null, attrValues);
    assertThat(substitutions)
        .containsExactlyElementsIn(Arrays.asList(expectedSubstitutions));
  }

  @Test
  public void testSingleScalarElementSubstitution() throws Exception {
    assertPlaceholderSubtitution(
        "%{x}",
        attrs(ImmutableMap.of("x", ImmutableList.of("a"))),
        new String[] {"a"},
        new String[] {"x"});
  }

  @Test
  public void testSingleVectorElementSubstitution() throws Exception {
    assertPlaceholderSubtitution(
        "%{x}",
        attrs(ImmutableMap.of("x", ImmutableList.of("a", "b", "c"))),
        new String[] {"a", "b", "c"},
        new String[] {"x"});
  }

  @Test
  public void testMultipleElementsSubstitution() throws Exception {
    assertPlaceholderSubtitution(
        "%{x}-%{y}-%{z}",
        attrs(
            ImmutableMap.of(
                "x", ImmutableList.of("foo", "bar", "baz"),
                "y", ImmutableList.of("meow"),
                "z", ImmutableList.of("1", "2"))),
        new String[] {
          "foo-meow-1", "foo-meow-2", "bar-meow-1", "bar-meow-2", "baz-meow-1", "baz-meow-2"
        },
        new String[] {"x", "y", "z"});
  }

  @Test
  public void testEmptyElementSubstitution() throws Exception {
    assertPlaceholderSubtitution(
        "a-%{x}",
        attrs(ImmutableMap.of("x", ImmutableList.<String>of())),
        new String[0],
        new String[] {"x"});
  }

  @Test
  public void testSamePlaceholderMultipleTimes() throws Exception {
    assertPlaceholderSubtitution(
        "%{x}-%{y}-%{x}",
        attrs(ImmutableMap.of("x", ImmutableList.of("a", "b"), "y", ImmutableList.of("1", "2"))),
        new String[] {"a-1-a", "a-1-b", "a-2-a", "a-2-b", "b-1-a", "b-1-b", "b-2-a", "b-2-b"},
        new String[] {"x", "y", "x"});
  }

  @Test
  public void testRepeatingPlaceholderValue() throws Exception {
    assertPlaceholderSubtitution(
        "%{x}",
        attrs(ImmutableMap.of("x", ImmutableList.of("a", "a"))),
        new String[] {"a"},
        new String[] {"x"});
  }

  @Test
  public void testIncompletePlaceholderTreatedAsText() throws Exception {
    assertPlaceholderSubtitution(
        "%{x}-%{y-%{z",
        attrs(ImmutableMap.of("x", ImmutableList.of("a", "b"))),
        new String[] {"a-%{y-%{z", "b-%{y-%{z"},
        new String[] {"x"});
  }
}
