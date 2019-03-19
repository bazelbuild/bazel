// Copyright 2006 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test of type-conversions using Type.
 */
@RunWith(JUnit4.class)
public class TypeTest {

  private Label currentRule;

  @Before
  public final void setCurrentRule() throws Exception  {
    this.currentRule = Label.parseAbsolute("//quux:baz", ImmutableMap.of());
  }

  @Test
  public void testInteger() throws Exception {
    Object x = 3;
    assertThat(Type.INTEGER.convert(x, null)).isEqualTo(x);
    assertThat(collectLabels(Type.INTEGER, x)).isEmpty();
  }

  @Test
  public void testNonInteger() throws Exception {
    try {
      Type.INTEGER.convert("foo", null);
      fail();
    } catch (Type.ConversionException e) {
      // This does not use assertMessageContainsWordsWithQuotes because at least
      // one test should test exact wording (but they all shouldn't to make
      // changing/improving the messages easy).
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("expected value of type 'int', but got \"foo\" (string)");
    }
  }

  // Ensure that types are reported correctly.
  @Test
  public void testTypeErrorMessage() throws Exception {
    try {
      Type.STRING_LIST.convert("[(1,2), 3, 4]", "myexpr", null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "expected value of type 'list(string)' for myexpr, "
                  + "but got \"[(1,2), 3, 4]\" (string)");
    }
  }

  @Test
  public void testString() throws Exception {
    Object s = "foo";
    assertThat(Type.STRING.convert(s, null)).isEqualTo(s);
    assertThat(collectLabels(Type.STRING, s)).isEmpty();
  }

  @Test
  public void testNonString() throws Exception {
    try {
      Type.STRING.convert(3, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessageThat().isEqualTo("expected value of type 'string', but got 3 (int)");
    }
  }

  @Test
  public void testBoolean() throws Exception {
    Object myTrue = true;
    Object myFalse = false;
    assertThat(Type.BOOLEAN.convert(1, null)).isEqualTo(Boolean.TRUE);
    assertThat(Type.BOOLEAN.convert(0, null)).isEqualTo(Boolean.FALSE);
    assertThat(Type.BOOLEAN.convert(true, null)).isTrue();
    assertThat(Type.BOOLEAN.convert(myTrue, null)).isTrue();
    assertThat(Type.BOOLEAN.convert(false, null)).isFalse();
    assertThat(Type.BOOLEAN.convert(myFalse, null)).isFalse();
    assertThat(collectLabels(Type.BOOLEAN, myTrue)).isEmpty();
  }

  @Test
  public void testNonBoolean() throws Exception {
    try {
      Type.BOOLEAN.convert("unexpected", null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("expected value of type 'int', but got \"unexpected\" (string)");
    }
    // Integers other than [0, 1] should fail.
    try {
      Type.BOOLEAN.convert(2, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessageThat().isEqualTo("boolean is not one of [0, 1]");
    }
    try {
      Type.BOOLEAN.convert(-1, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessageThat().isEqualTo("boolean is not one of [0, 1]");
    }
  }

  @Test
  public void testTriState() throws Exception {
    assertThat(BuildType.TRISTATE.convert(1, null)).isEqualTo(TriState.YES);
    assertThat(BuildType.TRISTATE.convert(0, null)).isEqualTo(TriState.NO);
    assertThat(BuildType.TRISTATE.convert(-1, null)).isEqualTo(TriState.AUTO);
    assertThat(BuildType.TRISTATE.convert(TriState.YES, null)).isEqualTo(TriState.YES);
    assertThat(BuildType.TRISTATE.convert(TriState.NO, null)).isEqualTo(TriState.NO);
    assertThat(BuildType.TRISTATE.convert(TriState.AUTO, null)).isEqualTo(TriState.AUTO);
    assertThat(collectLabels(BuildType.TRISTATE, TriState.YES)).isEmpty();

    // deprecated:
    assertThat(BuildType.TRISTATE.convert(true, null)).isEqualTo(TriState.YES);
    assertThat(BuildType.TRISTATE.convert(false, null)).isEqualTo(TriState.NO);
  }

  @Test
  public void testTriStateDoesNotAcceptArbitraryIntegers() throws Exception {
    List<Integer> listOfCases = Lists.newArrayList(2, 3, -5, -2, 20);
    for (Object entry : listOfCases) {
      try {
        BuildType.TRISTATE.convert(entry, null);
        fail();
      } catch (Type.ConversionException e) {
        // Expected.
      }
    }
  }

  @Test
  public void testTriStateDoesNotAcceptStrings() throws Exception {
    List<?> listOfCases = Lists.newArrayList("bad", "true", "auto", "false");
    // TODO(adonovan): add booleans true, false to this list; see b/116691720.
    for (Object entry : listOfCases) {
      try {
        BuildType.TRISTATE.convert(entry, null);
        fail();
      } catch (Type.ConversionException e) {
        // Expected.
      }
    }
  }

  @Test
  public void testTagConversion() throws Exception {
    assertThat(Type.BOOLEAN.toTagSet(true, "attribute"))
        .containsExactlyElementsIn(Sets.newHashSet("attribute"));
    assertThat(Type.BOOLEAN.toTagSet(false, "attribute"))
        .containsExactlyElementsIn(Sets.newHashSet("noattribute"));

    assertThat(Type.STRING.toTagSet("whiskey", "preferred_cocktail"))
        .containsExactlyElementsIn(Sets.newHashSet("whiskey"));

    assertThat(
            Type.STRING_LIST.toTagSet(
                Lists.newArrayList("cheddar", "ementaler", "gruyere"), "cheeses"))
        .containsExactlyElementsIn(Sets.newHashSet("cheddar", "ementaler", "gruyere"));
  }

  @Test
  public void testIllegalTagConversionByType() throws Exception {
    try {
      BuildType.TRISTATE.toTagSet(TriState.AUTO, "some_tristate");
      fail("Expect UnsuportedOperationException");
    } catch (UnsupportedOperationException e) {
      // Success.
    }
    try {
      BuildType.LICENSE.toTagSet(License.NO_LICENSE, "output_license");
      fail("Expect UnsuportedOperationException");
    } catch (UnsupportedOperationException e) {
      // Success.
    }
  }

  @Test
  public void testIllegalTagConversIonFromNullOnSupportedType() throws Exception {
    try {
      Type.BOOLEAN.toTagSet(null, "a_boolean");
      fail("Expect UnsuportedOperationException");
    } catch (IllegalStateException e) {
      // Success.
    }
  }

  @Test
  public void testLabel() throws Exception {
    Label label = Label.parseAbsolute("//foo:bar", ImmutableMap.of());
    assertThat(BuildType.LABEL.convert("//foo:bar", null, currentRule)).isEqualTo(label);
    assertThat(collectLabels(BuildType.LABEL, label)).containsExactly(label);
  }

  @Test
  public void testNodepLabel() throws Exception {
    Label label = Label.parseAbsolute("//foo:bar", ImmutableMap.of());
    assertThat(BuildType.NODEP_LABEL.convert("//foo:bar", null, currentRule)).isEqualTo(label);
    assertThat(collectLabels(BuildType.NODEP_LABEL, label)).containsExactly(label);
  }

  @Test
  public void testRelativeLabel() throws Exception {
    assertThat(BuildType.LABEL.convert(":wiz", null, currentRule))
        .isEqualTo(Label.parseAbsolute("//quux:wiz", ImmutableMap.of()));
    assertThat(BuildType.LABEL.convert("wiz", null, currentRule))
        .isEqualTo(Label.parseAbsolute("//quux:wiz", ImmutableMap.of()));
    try {
      BuildType.LABEL.convert("wiz", null);
      fail();
    } catch (ConversionException e) {
      /* ok */
    }
  }

  @Test
  public void testInvalidLabel() throws Exception {
    try {
      BuildType.LABEL.convert("not//a label", null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      MoreAsserts.assertContainsWordsWithQuotes(e.getMessage(), "not//a label");
    }
  }

  @Test
  public void testNonLabel() throws Exception {
    try {
      BuildType.LABEL.convert(3, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessageThat().isEqualTo("expected value of type 'string', but got 3 (int)");
    }
  }

  @Test
  public void testStringList() throws Exception {
    Object input = Arrays.asList("foo", "bar", "wiz");
    List<String> converted =
        Type.STRING_LIST.convert(input, null);
    assertThat(converted).isEqualTo(input);
    assertThat(converted).isNotSameAs(input);
    assertThat(collectLabels(Type.STRING_LIST, input)).isEmpty();
  }

  @Test
  public void testStringDict() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", "bang");
    Map<String, String> converted = Type.STRING_DICT.convert(input, null);
    assertThat(converted).isEqualTo(input);
    assertThat(converted).isNotSameAs(input);
    assertThat(collectLabels(Type.STRING_DICT, converted)).isEmpty();
  }

  @Test
  public void testStringDictBadElements() throws Exception {
    Object input = ImmutableMap.of("foo", MutableList.of(null, "bar", "baz"), "wiz", "bang");
    try {
      Type.STRING_DICT.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "expected value of type 'string' for dict value element, "
                  + "but got [\"bar\", \"baz\"] (list)");
    }
  }

  @Test
  public void testNonStringList() throws Exception {
    try {
      Type.STRING_LIST.convert(3, "blah");
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("expected value of type 'list(string)' for blah, but got 3 (int)");
    }
  }

  @Test
  public void testStringListBadElements() throws Exception {
    Object input = Arrays.<Object>asList("foo", "bar", 1);
    try {
      Type.STRING_LIST.convert(input, "argument quux");
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "expected value of type 'string' for element 2 of argument quux, but got 1 (int)");
    }
  }

  @Test
  public void testListDepsetConversion() throws Exception {
    Object input = SkylarkNestedSet.of(
        String.class,
        NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b", "c"));
    Type.STRING_LIST.convert(input, null);
  }

  @Test
  public void testLabelList() throws Exception {
    Object input = Arrays.asList("//foo:bar", ":wiz");
    List<Label> converted =
      BuildType.LABEL_LIST.convert(input , null, currentRule);
    List<Label> expected =
        Arrays.asList(
            Label.parseAbsolute("//foo:bar", ImmutableMap.of()),
            Label.parseAbsolute("//quux:wiz", ImmutableMap.of()));
    assertThat(converted).isEqualTo(expected);
    assertThat(converted).isNotSameAs(expected);
    assertThat(collectLabels(BuildType.LABEL_LIST, converted)).containsExactlyElementsIn(expected);
  }

  @Test
  public void testNonLabelList() throws Exception {
    try {
      BuildType.LABEL_LIST.convert(3, "foo", currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("expected value of type 'list(label)' for foo, but got 3 (int)");
    }
  }

  @Test
  public void testLabelListBadElements() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar", 2, "foo");
    try {
      BuildType.LABEL_LIST.convert(list, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("expected value of type 'string' for element 1 of null, but got 2 (int)");
    }
  }

  @Test
  public void testLabelListSyntaxError() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar/..", "foo");
    try {
      BuildType.LABEL_LIST.convert(list, "myexpr", currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "invalid label '//foo:bar/..' in element 0 of myexpr: "
                  + "invalid target name 'bar/..': "
                  + "target names may not contain up-level references '..'");
    }
  }

  @Test
  public void testStringListDict() throws Exception {
    Object input = ImmutableMap.of("foo", Arrays.asList("foo", "bar"),
                                   "wiz", Arrays.asList("bang"));
    Map<String, List<String>> converted =
        Type.STRING_LIST_DICT.convert(input, null, currentRule);
    Map<?, ?> expected = ImmutableMap.<String, List<String>>of(
            "foo", Arrays.asList("foo", "bar"),
            "wiz", Arrays.asList("bang"));
    assertThat(converted).isEqualTo(expected);
    assertThat(converted).isNotSameAs(expected);
    assertThat(collectLabels(Type.STRING_LIST_DICT, converted)).isEmpty();
  }

  @Test
  public void testStringListDictBadFirstElement() throws Exception {
    Object input = ImmutableMap.of(2, Arrays.asList("foo", "bar"),
                                   "wiz", Arrays.asList("bang"));
    try {
      Type.STRING_LIST_DICT.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("expected value of type 'string' for dict key element, but got 2 (int)");
    }
  }

  @Test
  public void testStringListDictBadSecondElement() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", Arrays.asList("bang"));
    try {
      Type.STRING_LIST_DICT.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "expected value of type 'list(string)' for dict value element, "
                  + "but got \"bar\" (string)");
    }
  }

  @Test
  public void testStringListDictBadElements1() throws Exception {
    Object input = ImmutableMap.of(Tuple.of("foo"), Tuple.of("bang"), "wiz", Tuple.of("bang"));
    try {
      Type.STRING_LIST_DICT.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "expected value of type 'string' for dict key element, but got "
                  + "(\"foo\",) (tuple)");
    }
  }

  @Test
  public void testStringDictThrowsConversionException() throws Exception {
    try {
      Type.STRING_DICT.convert("some string", null);
      fail();
    } catch (ConversionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              "expected value of type 'dict(string, string)', but got \"some string\" (string)");
    }
  }

  private static ImmutableList<Label> collectLabels(Type<?> type, Object value) {
    final ImmutableList.Builder<Label> result = ImmutableList.builder();
    type.visitLabels((label, dummy) -> result.add(label), value, /*context=*/ null);
    return result.build();
  }
}
