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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import com.google.devtools.build.lib.testutil.MoreAsserts;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Test of type-conversions using Type.
 */
@RunWith(JUnit4.class)
public class TypeTest {

  private Label currentRule;

  @Before
  public final void setCurrentRule() throws Exception  {
    this.currentRule = Label.parseAbsolute("//quux:baz");
  }

  @Test
  public void testInteger() throws Exception {
    Object x = 3;
    assertEquals(x, Type.INTEGER.convert(x, null));
    assertThat(Type.INTEGER.flatten(x)).isEmpty();
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
      assertThat(e).hasMessage("expected value of type 'int', but got \"foo\" (string)");
    }
  }

  // Ensure that types are reported correctly.
  @Test
  public void testTypeErrorMessage() throws Exception {
    try {
      Type.STRING_LIST.convert("[(1,2), 3, 4]", "myexpr", null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'list(string)' for myexpr, "
          + "but got \"[(1,2), 3, 4]\" (string)");
    }
  }

  @Test
  public void testString() throws Exception {
    Object s = "foo";
    assertEquals(s, Type.STRING.convert(s, null));
    assertThat(Type.STRING.flatten(s)).isEmpty();
  }

  @Test
  public void testNonString() throws Exception {
    try {
      Type.STRING.convert(3, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string', but got 3 (int)");
    }
  }

  @Test
  public void testBoolean() throws Exception {
    Object myTrue = true;
    Object myFalse = false;
    assertEquals(Boolean.TRUE, Type.BOOLEAN.convert(1, null));
    assertEquals(Boolean.FALSE, Type.BOOLEAN.convert(0, null));
    assertTrue(Type.BOOLEAN.convert(true, null));
    assertTrue(Type.BOOLEAN.convert(myTrue, null));
    assertFalse(Type.BOOLEAN.convert(false, null));
    assertFalse(Type.BOOLEAN.convert(myFalse, null));
    assertThat(Type.BOOLEAN.flatten(myTrue)).isEmpty();
  }

  @Test
  public void testNonBoolean() throws Exception {
    try {
      Type.BOOLEAN.convert("unexpected", null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage(
          "expected value of type 'int', but got \"unexpected\" (string)");
    }
    // Integers other than [0, 1] should fail.
    try {
      Type.BOOLEAN.convert(2, null);
      fail();
    } catch (Type.ConversionException e) {
      assertEquals(e.getMessage(), "boolean is not one of [0, 1]");
    }
    try {
      Type.BOOLEAN.convert(-1, null);
      fail();
    } catch (Type.ConversionException e) {
      assertEquals(e.getMessage(), "boolean is not one of [0, 1]");
    }
  }

  @Test
  public void testTriState() throws Exception {
    Assert.assertEquals(TriState.YES, BuildType.TRISTATE.convert(1, null));
    assertEquals(TriState.NO, BuildType.TRISTATE.convert(0, null));
    assertEquals(TriState.AUTO, BuildType.TRISTATE.convert(-1, null));
    assertEquals(TriState.YES, BuildType.TRISTATE.convert(true, null));
    assertEquals(TriState.NO, BuildType.TRISTATE.convert(false, null));
    assertEquals(TriState.YES, BuildType.TRISTATE.convert(TriState.YES, null));
    assertEquals(TriState.NO, BuildType.TRISTATE.convert(TriState.NO, null));
    assertEquals(TriState.AUTO, BuildType.TRISTATE.convert(TriState.AUTO, null));
    assertThat(BuildType.TRISTATE.flatten(TriState.YES)).isEmpty();
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
    List<String> listOfCases = Lists.newArrayList("bad", "true", "auto", "false");
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
    Label label = Label
        .parseAbsolute("//foo:bar");
    assertEquals(label, BuildType.LABEL.convert("//foo:bar", null, currentRule));
    assertThat(BuildType.LABEL.flatten(label)).containsExactly(label);
  }

  @Test
  public void testNodepLabel() throws Exception {
    Label label = Label
        .parseAbsolute("//foo:bar");
    assertEquals(label, BuildType.NODEP_LABEL.convert("//foo:bar", null, currentRule));
    assertThat(BuildType.NODEP_LABEL.flatten(label)).containsExactly(label);
  }

  @Test
  public void testRelativeLabel() throws Exception {
    assertEquals(Label.parseAbsolute("//quux:wiz"),
        BuildType.LABEL.convert(":wiz", null, currentRule));
    assertEquals(Label.parseAbsolute("//quux:wiz"),
        BuildType.LABEL.convert("wiz", null, currentRule));
    try {
      BuildType.LABEL.convert("wiz", null);
      fail();
    } catch (NullPointerException e) {
      /* ok */
    }
  }

  @Test
  public void testInvalidLabel() throws Exception {
    try {
      BuildType.LABEL.convert("not a label", null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      MoreAsserts.assertContainsWordsWithQuotes(e.getMessage(), "not a label");
    }
  }

  @Test
  public void testNonLabel() throws Exception {
    try {
      BuildType.LABEL.convert(3, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string', but got 3 (int)");
    }
  }

  @Test
  public void testStringList() throws Exception {
    Object input = Arrays.asList("foo", "bar", "wiz");
    List<String> converted =
        Type.STRING_LIST.convert(input, null);
    assertEquals(input, converted);
    assertNotSame(input, converted);
    assertThat(Type.STRING_LIST.flatten(input)).isEmpty();
  }

  @Test
  public void testStringDict() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", "bang");
    Map<String, String> converted = Type.STRING_DICT.convert(input, null);
    assertEquals(input, converted);
    assertNotSame(input, converted);
    assertThat(Type.STRING_DICT.flatten(converted)).isEmpty();
  }

  @Test
  public void testStringDictBadElements() throws Exception {
    Object input = ImmutableMap.of("foo", MutableList.of(null, "bar", "baz"), "wiz", "bang");
    try {
      Type.STRING_DICT.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict value element, "
          + "but got [\"bar\", \"baz\"] (list)");
    }
  }

  @Test
  public void testNonStringList() throws Exception {
    try {
      Type.STRING_LIST.convert(3, "blah");
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'list(string)' for blah, but got 3 (int)");
    }
  }

  @Test
  public void testStringListBadElements() throws Exception {
    Object input = Arrays.<Object>asList("foo", "bar", 1);
    try {
      Type.STRING_LIST.convert(input, "argument quux");
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage(
          "expected value of type 'string' for element 2 of argument quux, but got 1 (int)");
    }
  }

  @Test
  public void testLabelList() throws Exception {
    Object input = Arrays.asList("//foo:bar", ":wiz");
    List<Label> converted =
      BuildType.LABEL_LIST.convert(input , null, currentRule);
    List<Label> expected =
      Arrays.asList(Label.parseAbsolute("//foo:bar"),
                    Label.parseAbsolute("//quux:wiz"));
    assertEquals(expected, converted);
    assertNotSame(expected, converted);
    assertThat(BuildType.LABEL_LIST.flatten(converted)).containsExactlyElementsIn(expected);
  }

  @Test
  public void testNonLabelList() throws Exception {
    try {
      BuildType.LABEL_LIST.convert(3, "foo", currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'list(label)' for foo, but got 3 (int)");
    }
  }

  @Test
  public void testLabelListBadElements() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar", 2, "foo");
    try {
      BuildType.LABEL_LIST.convert(list, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage(
          "expected value of type 'string' for element 1 of null, but got 2 (int)");
    }
  }

  @Test
  public void testLabelListSyntaxError() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar/..", "foo");
    try {
      BuildType.LABEL_LIST.convert(list, "myexpr", currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("invalid label '//foo:bar/..' in element 0 of myexpr: "
          + "invalid target name 'bar/..': "
          + "target names may not contain up-level references '..'");
    }
  }

  @Test
  public void testLabelListDict() throws Exception {
    Object input = ImmutableMap.of("foo", Arrays.asList("//foo:bar"),
        "wiz", Arrays.asList(":bang"));
    Map<String, List<Label>> converted =
        BuildType.LABEL_LIST_DICT.convert(input, null, currentRule);
    Label fooLabel = Label
        .parseAbsolute("//foo:bar");
    Label bangLabel = Label
        .parseAbsolute("//quux:bang");
    Map<?, ?> expected = ImmutableMap.<String, List<Label>>of(
            "foo", Arrays.<Label>asList(fooLabel),
            "wiz", Arrays.<Label>asList(bangLabel));
    assertEquals(expected, converted);
    assertNotSame(expected, converted);
    assertThat(BuildType.LABEL_LIST_DICT.flatten(converted)).containsExactly(fooLabel, bangLabel);
  }

  @Test
  public void testLabelListDictBadFirstElement() throws Exception {
    Object input = ImmutableMap.of(2, Arrays.asList("//foo:bar"),
        "wiz", Arrays.asList(":bang"));
    try {
      BuildType.LABEL_LIST_DICT.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage(
          "expected value of type 'string' for dict key element, but got 2 (int)");
    }
  }

  @Test
  public void testLabelListDictBadSecondElement() throws Exception {
    Object input = ImmutableMap.of("foo", "//foo:bar",
                                   "wiz", Arrays.asList(":bang"));
    try {
      BuildType.LABEL_LIST_DICT.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage(
          "expected value of type 'list(label)' for dict value element, "
          + "but got \"//foo:bar\" (string)");
    }
  }

  @Test
  public void testLabelListDictBadElements1() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "bar", Arrays.asList("//foo:bar"),
                                   "wiz", Arrays.asList(":bang"));
    try {
      BuildType.LABEL_LIST_DICT.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'list(label)' for dict value element, "
          + "but got \"bar\" (string)");
    }
  }

  @Test
  public void testLabelListDictSyntaxError() throws Exception {
    Object input = ImmutableMap.of("foo", Arrays.asList("//foo:.."),
                                   "wiz", Arrays.asList(":bang"));
    try {
      BuildType.LABEL_LIST_DICT.convert(input, "baz", currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("invalid label '//foo:..' in element 0 of dict value element: "
          + "invalid target name '..': "
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
    assertEquals(expected, converted);
    assertNotSame(expected, converted);
    assertThat(Type.STRING_LIST_DICT.flatten(converted)).isEmpty();
  }

  @Test
  public void testStringListDictBadFirstElement() throws Exception {
    Object input = ImmutableMap.of(2, Arrays.asList("foo", "bar"),
                                   "wiz", Arrays.asList("bang"));
    try {
      Type.STRING_LIST_DICT.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage(
          "expected value of type 'string' for dict key element, but got 2 (int)");
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
      assertThat(e).hasMessage(
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
      assertThat(e).hasMessage("expected value of type 'string' for dict key element, but got "
          + "(\"foo\",) (tuple)");
    }
  }

  @Test
  public void testStringDictUnary() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", "bang");
    Map<?, ?> converted =
        Type.STRING_DICT_UNARY.convert(input, null, currentRule);
    Map<?, ?> expected = ImmutableMap.<String, String>of(
            "foo", "bar",
            "wiz", "bang");
    assertEquals(expected, converted);
    assertNotSame(expected, converted);
    assertThat(Type.STRING_DICT_UNARY.flatten(converted)).isEmpty();
  }

  @Test
  public void testStringDictUnaryBadFirstElement() throws Exception {
    Object input = ImmutableMap.of(2, Arrays.asList("foo", "bar"),
                                   "wiz", Arrays.asList("bang"));
    try {
      Type.STRING_DICT_UNARY.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict key element, but got "
          + "2 (int)");
    }
  }

  @Test
  public void testStringDictUnaryBadSecondElement() throws Exception {
    Object input = ImmutableMap.of("foo", "bar", "wiz", MutableList.of(null, "bang"));
    try {
      Type.STRING_DICT_UNARY.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict value element, but got "
          + "[\"bang\"] (list)");
    }
  }

  @Test
  public void testStringDictUnaryBadElements1() throws Exception {
    Object input = ImmutableMap.of("foo", "bar", Tuple.of("foo", "bar"), Tuple.of("wiz", "bang"));
    try {
      Type.STRING_DICT_UNARY.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict key element, but got "
          + "(\"foo\", \"bar\") (tuple)");
    }
  }

  @Test
  public void testStringDictThrowsConversionException() throws Exception {
    try {
      Type.STRING_DICT.convert("some string", null);
      fail();
    } catch (ConversionException e) {
      assertThat(e).hasMessage("Expected a map for dictionary but got a java.lang.String");
    }
  }
}
