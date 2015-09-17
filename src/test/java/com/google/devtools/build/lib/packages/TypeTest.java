// Copyright 2006-2015 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertSameContents;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SelectorList;
import com.google.devtools.build.lib.syntax.SelectorValue;
import com.google.devtools.build.lib.testutil.MoreAsserts;

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
  public void setUp() throws Exception {
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
    assertEquals(TriState.YES, Type.TRISTATE.convert(1, null));
    assertEquals(TriState.NO, Type.TRISTATE.convert(0, null));
    assertEquals(TriState.AUTO, Type.TRISTATE.convert(-1, null));
    assertEquals(TriState.YES, Type.TRISTATE.convert(true, null));
    assertEquals(TriState.NO, Type.TRISTATE.convert(false, null));
    assertEquals(TriState.YES, Type.TRISTATE.convert(TriState.YES, null));
    assertEquals(TriState.NO, Type.TRISTATE.convert(TriState.NO, null));
    assertEquals(TriState.AUTO, Type.TRISTATE.convert(TriState.AUTO, null));
    assertThat(Type.TRISTATE.flatten(TriState.YES)).isEmpty();
  }

  @Test
  public void testTriStateDoesNotAcceptArbitraryIntegers() throws Exception {
    List<Integer> listOfCases = Lists.newArrayList(2, 3, -5, -2, 20);
    for (Object entry : listOfCases) {
      try {
        Type.TRISTATE.convert(entry, null);
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
        Type.TRISTATE.convert(entry, null);
        fail();
      } catch (Type.ConversionException e) {
        // Expected.
      }
    }
  }

  @Test
  public void testTagConversion() throws Exception {
    assertSameContents(Sets.newHashSet("attribute"),
        Type.BOOLEAN.toTagSet(true, "attribute"));
    assertSameContents(Sets.newHashSet("noattribute"),
        Type.BOOLEAN.toTagSet(false, "attribute"));

    assertSameContents(Sets.newHashSet("whiskey"),
        Type.STRING.toTagSet("whiskey", "preferred_cocktail"));

    assertSameContents(Sets.newHashSet("cheddar", "ementaler", "gruyere"),
        Type.STRING_LIST.toTagSet(
            Lists.newArrayList("cheddar", "ementaler", "gruyere"), "cheeses"));
  }

  @Test
  public void testIllegalTagConversionByType() throws Exception {
    try {
      Type.TRISTATE.toTagSet(TriState.AUTO, "some_tristate");
      fail("Expect UnsuportedOperationException");
    } catch (UnsupportedOperationException e) {
      // Success.
    }
    try {
      Type.LICENSE.toTagSet(License.NO_LICENSE, "output_license");
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
    Label label = Label.parseAbsolute("//foo:bar");
    assertEquals(label, Type.LABEL.convert("//foo:bar", null, currentRule));
    assertThat(Type.LABEL.flatten(label)).containsExactly(label);
  }

  @Test
  public void testNodepLabel() throws Exception {
    Label label = Label.parseAbsolute("//foo:bar");
    assertEquals(label, Type.NODEP_LABEL.convert("//foo:bar", null, currentRule));
    assertThat(Type.NODEP_LABEL.flatten(label)).containsExactly(label);
  }

  @Test
  public void testRelativeLabel() throws Exception {
    assertEquals(Label.parseAbsolute("//quux:wiz"),
        Type.LABEL.convert(":wiz", null, currentRule));
    assertEquals(Label.parseAbsolute("//quux:wiz"),
        Type.LABEL.convert("wiz", null, currentRule));
    try {
      Type.LABEL.convert("wiz", null);
      fail();
    } catch (NullPointerException e) {
      /* ok */
    }
  }

  @Test
  public void testInvalidLabel() throws Exception {
    try {
      Type.LABEL.convert("not a label", null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      MoreAsserts.assertContainsWordsWithQuotes(e.getMessage(), "not a label");
    }
  }

  @Test
  public void testNonLabel() throws Exception {
    try {
      Type.LABEL.convert(3, null);
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
    Object input = ImmutableMap.of("foo", Arrays.asList("bar", "baz"),
        "wiz", "bang");
    try {
      Type.STRING_DICT.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict value element, "
          + "but got [\"bar\", \"baz\"] (List)");
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
      Type.LABEL_LIST.convert(input , null, currentRule);
    List<Label> expected =
      Arrays.asList(Label.parseAbsolute("//foo:bar"),
                    Label.parseAbsolute("//quux:wiz"));
    assertEquals(expected, converted);
    assertNotSame(expected, converted);
    assertThat(Type.LABEL_LIST.flatten(converted)).containsExactlyElementsIn(expected);
  }

  @Test
  public void testNonLabelList() throws Exception {
    try {
      Type.LABEL_LIST.convert(3, "foo", currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'list(label)' for foo, but got 3 (int)");
    }
  }

  @Test
  public void testLabelListBadElements() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar", 2, "foo");
    try {
      Type.LABEL_LIST.convert(list, null, currentRule);
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
      Type.LABEL_LIST.convert(list, "myexpr", currentRule);
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
    Map<String, List<Label>> converted = Type.LABEL_LIST_DICT.convert(input, null, currentRule);
    Label fooLabel = Label.parseAbsolute("//foo:bar");
    Label bangLabel = Label.parseAbsolute("//quux:bang");
    Map<?, ?> expected = ImmutableMap.<String, List<Label>>of(
            "foo", Arrays.<Label>asList(fooLabel),
            "wiz", Arrays.<Label>asList(bangLabel));
    assertEquals(expected, converted);
    assertNotSame(expected, converted);
    assertThat(Type.LABEL_LIST_DICT.flatten(converted)).containsExactly(fooLabel, bangLabel);
  }

  @Test
  public void testLabelListDictBadFirstElement() throws Exception {
    Object input = ImmutableMap.of(2, Arrays.asList("//foo:bar"),
        "wiz", Arrays.asList(":bang"));
    try {
      Type.LABEL_LIST_DICT.convert(input, null, currentRule);
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
      Type.LABEL_LIST_DICT.convert(input, null, currentRule);
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
      Type.LABEL_LIST_DICT.convert(input, null);
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
      Type.LABEL_LIST_DICT.convert(input, "baz", currentRule);
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
    Object input = ImmutableMap.of(Arrays.asList("foo"), Arrays.asList("bang"),
                                   "wiz", Arrays.asList("bang"));
    try {
      Type.STRING_LIST_DICT.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict key element, but got "
          + "[\"foo\"] (List)");
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
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", Arrays.asList("bang"));
    try {
      Type.STRING_DICT_UNARY.convert(input, null, currentRule);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict value element, but got "
          + "[\"bang\"] (List)");
    }
  }

  @Test
  public void testStringDictUnaryBadElements1() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   Arrays.asList("foo", "bar"),
                                   Arrays.<Object>asList("wiz", "bang"));
    try {
      Type.STRING_DICT_UNARY.convert(input, null);
      fail();
    } catch (Type.ConversionException e) {
      assertThat(e).hasMessage("expected value of type 'string' for dict key element, but got "
          + "[\"foo\", \"bar\"] (List)");
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

  @Test
  public void testFilesetEntry() throws Exception {
    Label srcDir = Label.create("foo", "src");
    Label entryLabel = Label.create("foo", "entry");
    FilesetEntry input =
        new FilesetEntry(srcDir, ImmutableList.of(entryLabel), null, null, null, null);
    assertEquals(input, Type.FILESET_ENTRY.convert(input, null, currentRule));
    assertThat(Type.FILESET_ENTRY.flatten(input)).containsExactly(entryLabel);
  }

  @Test
  public void testFilesetEntryList() throws Exception {
    Label srcDir = Label.create("foo", "src");
    Label entry1Label = Label.create("foo", "entry1");
    Label entry2Label = Label.create("foo", "entry");
    List<FilesetEntry> input = ImmutableList.of(
        new FilesetEntry(srcDir, ImmutableList.of(entry1Label), null, null, null, null),
        new FilesetEntry(srcDir, ImmutableList.of(entry2Label), null, null, null, null));
    assertEquals(input, Type.FILESET_ENTRY_LIST.convert(input, null, currentRule));
    assertThat(Type.FILESET_ENTRY_LIST.flatten(input)).containsExactly(entry1Label, entry2Label);
  }

  /**
   * Tests basic {@link Type.Selector} functionality.
   */
  @Test
  public void testSelector() throws Exception {
    Object input = ImmutableMap.of(
        "//conditions:a", "//a:a",
        "//conditions:b", "//b:b",
        Type.Selector.DEFAULT_CONDITION_KEY, "//d:d");
    Type.Selector<Label> selector = new Type.Selector<>(input, null, currentRule, Type.LABEL);
    assertEquals(Type.LABEL, selector.getOriginalType());

    Map<Label, Label> expectedMap = ImmutableMap.of(
        Label.parseAbsolute("//conditions:a"), Label.create("a", "a"),
        Label.parseAbsolute("//conditions:b"), Label.create("b", "b"),
        Label.parseAbsolute(Type.Selector.DEFAULT_CONDITION_KEY), Label.create("d", "d"));
    assertSameContents(expectedMap.entrySet(), selector.getEntries().entrySet());
  }

  /**
   * Tests that creating a {@link Type.Selector} over a mismatching native type triggers an
   * exception.
   */
  @Test
  public void testSelectorWrongType() throws Exception {
    Object input = ImmutableMap.of(
        "//conditions:a", "not a label",
        "//conditions:b", "also not a label",
        Type.Selector.DEFAULT_CONDITION_KEY, "whatever");
    try {
      new Type.Selector<Label>(input, null, currentRule, Type.LABEL);
      fail("Expected Selector instantiation to fail since the input isn't a selection of labels");
    } catch (ConversionException e) {
      assertThat(e.getMessage()).contains("invalid label 'not a label'");
    }
  }

  /**
   * Tests that non-label selector keys trigger an exception.
   */
  @Test
  public void testSelectorKeyIsNotALabel() throws Exception {
    Object input = ImmutableMap.of(
        "not a label", "//a:a",
        Type.Selector.DEFAULT_CONDITION_KEY, "whatever");
    try {
      new Type.Selector<Label>(input, null, currentRule, Type.LABEL);
      fail("Expected Selector instantiation to fail since the key isn't a label");
    } catch (ConversionException e) {
      assertThat(e.getMessage()).contains("invalid label 'not a label'");
    }
  }

  /**
   * Tests that {@link Type.Selector} correctly references its default value.
   */
  @Test
  public void testSelectorDefault() throws Exception {
    Object input = ImmutableMap.of(
        "//conditions:a", "//a:a",
        "//conditions:b", "//b:b",
        Type.Selector.DEFAULT_CONDITION_KEY, "//d:d");
    assertEquals(
        Label.create("d", "d"),
        new Type.Selector<Label>(input, null, currentRule, Type.LABEL).getDefault());
  }

  @Test
  public void testSelectorList() throws Exception {
    Object selector1 = new SelectorValue(ImmutableMap.of("//conditions:a",
        ImmutableList.of("//a:a"), "//conditions:b", ImmutableList.of("//b:b")));
    Object selector2 = new SelectorValue(ImmutableMap.of("//conditions:c",
        ImmutableList.of("//c:c"), "//conditions:d", ImmutableList.of("//d:d")));
    Type.SelectorList<List<Label>> selectorList = new Type.SelectorList<>(
        ImmutableList.of(selector1, selector2), null, currentRule, Type.LABEL_LIST);

    assertEquals(Type.LABEL_LIST, selectorList.getOriginalType());
    assertSameContents(
        ImmutableSet.of(
            Label.parseAbsolute("//conditions:a"), Label.parseAbsolute("//conditions:b"),
            Label.parseAbsolute("//conditions:c"), Label.parseAbsolute("//conditions:d")),
        selectorList.getKeyLabels());

    List<Type.Selector<List<Label>>> selectors = selectorList.getSelectors();
    assertSameContents(
        ImmutableMap.of(
                Label.parseAbsolute("//conditions:a"), ImmutableList.of(Label.create("a", "a")),
                Label.parseAbsolute("//conditions:b"), ImmutableList.of(Label.create("b", "b")))
            .entrySet(),
        selectors.get(0).getEntries().entrySet());
    assertSameContents(
        ImmutableMap.of(
                Label.parseAbsolute("//conditions:c"), ImmutableList.of(Label.create("c", "c")),
                Label.parseAbsolute("//conditions:d"), ImmutableList.of(Label.create("d", "d")))
            .entrySet(),
        selectors.get(1).getEntries().entrySet());
  }

  @Test
  public void testSelectorListMixedTypes() throws Exception {
    Object selector1 =
        new SelectorValue(ImmutableMap.of("//conditions:a", ImmutableList.of("//a:a")));
    Object selector2 =
        new SelectorValue(ImmutableMap.of("//conditions:b", "//b:b"));
    try {
      new Type.SelectorList<>(ImmutableList.of(selector1, selector2), null, currentRule,
          Type.LABEL_LIST);
      fail("Expected SelectorList initialization to fail on mixed element types");
    } catch (ConversionException e) {
      assertThat(e.getMessage()).contains("expected value of type 'list(label)'");
    }
  }

  /**
   * Tests that {@link Type#selectableConvert} returns either the native type or a selector
   * on that type, in accordance with the provided input.
   */
  @SuppressWarnings("unchecked")
  @Test
  public void testSelectableConvert() throws Exception {
    Object nativeInput = Arrays.asList("//a:a1", "//a:a2");
    Object selectableInput =
        SelectorList.of(new SelectorValue(ImmutableMap.of(
            "//conditions:a", nativeInput,
            Type.Selector.DEFAULT_CONDITION_KEY, nativeInput)));
    List<Label> expectedLabels = ImmutableList.of(Label.create("a", "a1"), Label.create("a", "a2"));

    // Conversion to direct type:
    Object converted = Type.LABEL_LIST.selectableConvert(nativeInput, null, currentRule);
    assertTrue(converted instanceof List<?>);
    assertSameContents(expectedLabels, (List<Label>) converted);

    // Conversion to selectable type:
    converted = Type.LABEL_LIST.selectableConvert(selectableInput, null, currentRule);
    Type.SelectorList<?> selectorList = (Type.SelectorList<?>) converted;
    assertSameContents(
        ImmutableMap.of(
            Label.parseAbsolute("//conditions:a"), expectedLabels,
            Label.parseAbsolute(Type.Selector.DEFAULT_CONDITION_KEY), expectedLabels).entrySet(),
        ((Type.Selector<Label>) selectorList.getSelectors().get(0)).getEntries().entrySet());
  }

  /**
   * Tests that {@link Type#convert} fails on selector inputs.
   */
  @Test
  public void testConvertDoesNotAcceptSelectables() throws Exception {
    Object selectableInput = SelectorList.of(
        new SelectorValue(ImmutableMap.of("//conditions:a", Arrays.asList("//a:a1", "//a:a2"))));
    try {
      Type.LABEL_LIST.convert(selectableInput, null, currentRule);
      fail("Expected conversion to fail on a selectable input");
    } catch (ConversionException e) {
      assertThat(e.getMessage()).contains("expected value of type 'list(label)'");
    }
  }

  /**
   * Tests for "reserved" key labels (i.e. not intended to map to actual targets).
   */
  @Test
  public void testReservedKeyLabels() throws Exception {
    assertFalse(Type.Selector.isReservedLabel(Label.parseAbsolute("//condition:a")));
    assertTrue(Type.Selector.isReservedLabel(
        Label.parseAbsolute(Type.Selector.DEFAULT_CONDITION_KEY)));
  }
}
