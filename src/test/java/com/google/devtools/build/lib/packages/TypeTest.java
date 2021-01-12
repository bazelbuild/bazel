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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.Tuple;
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
    Object x = StarlarkInt.of(3);
    assertThat(Type.INTEGER.convert(x, null)).isEqualTo(x);
    assertThat(collectLabels(Type.INTEGER, x)).isEmpty();

    // INTEGER rule attributes must be in signed 32-bit value range.
    // (If we ever relax this, we'll need to audit every place that
    // converts an attribute to an int using toIntUnchecked, since
    // that operation might then fail, and extend the Package
    // serialization protocol to support bigint.)
    StarlarkInt big = StarlarkInt.of(111111111);
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> Type.INTEGER.convert(StarlarkInt.multiply(big, big), "param"));
    assertThat(e)
        .hasMessageThat()
        .contains("for param, got 12345678987654321, want value in signed 32-bit range");

    // Ensure that the range of INTEGER.concat is int32.
    assertThat(Type.INTEGER.concat(Arrays.asList(StarlarkInt.of(0x7fffffff), StarlarkInt.of(1))))
        .isEqualTo(StarlarkInt.of(-0x80000000));
  }

  @Test
  public void testNonInteger() throws Exception {
    Type.ConversionException e =
        assertThrows(Type.ConversionException.class, () -> Type.INTEGER.convert("foo", null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected value of type 'int', but got \"foo\" (string)");
  }

  // Ensure that types are reported correctly.
  @Test
  public void testTypeErrorMessage() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> Type.STRING_LIST.convert("[(1,2), 3, 4]", "myexpr", null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "expected value of type 'list(string)' for myexpr, "
                + "but got \"[(1,2), 3, 4]\" (string)");
  }

  @Test
  public void testString() throws Exception {
    Object s = "foo";
    assertThat(Type.STRING.convert(s, null)).isEqualTo(s);
    assertThat(collectLabels(Type.STRING, s)).isEmpty();
  }

  @Test
  public void testNonString() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class, () -> Type.STRING.convert(StarlarkInt.of(3), null));
    assertThat(e).hasMessageThat().isEqualTo("expected value of type 'string', but got 3 (int)");
  }

  @Test
  public void testBoolean() throws Exception {
    Object myTrue = true;
    Object myFalse = false;
    assertThat(Type.BOOLEAN.convert(StarlarkInt.of(1), null)).isEqualTo(Boolean.TRUE);
    assertThat(Type.BOOLEAN.convert(StarlarkInt.of(0), null)).isEqualTo(Boolean.FALSE);
    assertThat(Type.BOOLEAN.convert(true, null)).isTrue();
    assertThat(Type.BOOLEAN.convert(myTrue, null)).isTrue();
    assertThat(Type.BOOLEAN.convert(false, null)).isFalse();
    assertThat(Type.BOOLEAN.convert(myFalse, null)).isFalse();
    assertThat(collectLabels(Type.BOOLEAN, myTrue)).isEmpty();
  }

  @Test
  public void testNonBoolean() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class, () -> Type.BOOLEAN.convert("unexpected", null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected value of type 'int', but got \"unexpected\" (string)");
    // Integers other than [0, 1] should fail.
    e =
        assertThrows(
            Type.ConversionException.class, () -> Type.BOOLEAN.convert(StarlarkInt.of(2), null));
    assertThat(e).hasMessageThat().isEqualTo("boolean is not one of [0, 1]");
    e =
        assertThrows(
            Type.ConversionException.class, () -> Type.BOOLEAN.convert(StarlarkInt.of(-1), null));
    assertThat(e).hasMessageThat().isEqualTo("boolean is not one of [0, 1]");
  }

  @Test
  public void testTriState() throws Exception {
    assertThat(BuildType.TRISTATE.convert(StarlarkInt.of(1), null)).isEqualTo(TriState.YES);
    assertThat(BuildType.TRISTATE.convert(StarlarkInt.of(0), null)).isEqualTo(TriState.NO);
    assertThat(BuildType.TRISTATE.convert(StarlarkInt.of(-1), null)).isEqualTo(TriState.AUTO);
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
    for (Integer i : Lists.newArrayList(2, 3, -5, -2, 20)) {
      assertThrows(
          Type.ConversionException.class,
          () -> BuildType.TRISTATE.convert(StarlarkInt.of(i), null));
    }
  }

  @Test
  public void testTriStateDoesNotAcceptStrings() throws Exception {
    List<?> listOfCases = Lists.newArrayList("bad", "true", "auto", "false");
    // TODO(adonovan): add booleans true, false to this list; see b/116691720.
    for (Object entry : listOfCases) {
      assertThrows(Type.ConversionException.class, () -> BuildType.TRISTATE.convert(entry, null));
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
    assertThrows(
        UnsupportedOperationException.class,
        () -> BuildType.TRISTATE.toTagSet(TriState.AUTO, "some_tristate"));
    assertThrows(
        UnsupportedOperationException.class,
        () -> BuildType.LICENSE.toTagSet(License.NO_LICENSE, "output_license"));
  }

  @Test
  public void testIllegalTagConversIonFromNullOnSupportedType() throws Exception {
    assertThrows(IllegalStateException.class, () -> Type.BOOLEAN.toTagSet(null, "a_boolean"));
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
    assertThrows(Type.ConversionException.class, () -> BuildType.LABEL.convert("wiz", null));
  }

  @Test
  public void testInvalidLabel() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> BuildType.LABEL.convert("not//a label", null, currentRule));
    MoreAsserts.assertContainsWordsWithQuotes(e.getMessage(), "not//a label");
  }

  @Test
  public void testNonLabel() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class, () -> BuildType.LABEL.convert(StarlarkInt.of(3), null));
    assertThat(e).hasMessageThat().isEqualTo("expected value of type 'string', but got 3 (int)");
  }

  @Test
  public void testStringList() throws Exception {
    Object input = Arrays.asList("foo", "bar", "wiz");
    List<String> converted =
        Type.STRING_LIST.convert(input, null);
    assertThat(converted).isEqualTo(input);
    assertThat(converted).isNotSameInstanceAs(input);
    assertThat(collectLabels(Type.STRING_LIST, input)).isEmpty();
  }

  @Test
  public void testStringDict() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", "bang");
    Map<String, String> converted = Type.STRING_DICT.convert(input, null);
    assertThat(converted).isEqualTo(input);
    assertThat(converted).isNotSameInstanceAs(input);
    assertThat(collectLabels(Type.STRING_DICT, converted)).isEmpty();
  }

  @Test
  public void testStringDictBadElements() throws Exception {
    Object input = ImmutableMap.of("foo", StarlarkList.of(null, "bar", "baz"), "wiz", "bang");
    Type.ConversionException e =
        assertThrows(Type.ConversionException.class, () -> Type.STRING_DICT.convert(input, null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "expected value of type 'string' for dict value element, "
                + "but got [\"bar\", \"baz\"] (list)");
  }

  @Test
  public void testNonStringList() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> Type.STRING_LIST.convert(StarlarkInt.of(3), "blah"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected value of type 'list(string)' for blah, but got 3 (int)");
  }

  @Test
  public void testStringListBadElements() throws Exception {
    Object input = Arrays.<Object>asList("foo", "bar", StarlarkInt.of(1));
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class, () -> Type.STRING_LIST.convert(input, "argument quux"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "expected value of type 'string' for element 2 of argument quux, but got 1 (int)");
  }

  @Test
  public void testListDepsetConversion() throws Exception {
    Object input =
        Depset.of(
            Depset.ElementType.STRING, NestedSetBuilder.create(Order.STABLE_ORDER, "a", "b", "c"));
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
    assertThat(converted).isNotSameInstanceAs(expected);
    assertThat(collectLabels(BuildType.LABEL_LIST, converted)).containsExactlyElementsIn(expected);
  }

  @Test
  public void testNonLabelList() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> BuildType.LABEL_LIST.convert(StarlarkInt.of(3), "foo", currentRule));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected value of type 'list(label)' for foo, but got 3 (int)");
  }

  @Test
  public void testLabelListBadElements() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar", StarlarkInt.of(2), "foo");
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> BuildType.LABEL_LIST.convert(list, null, currentRule));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected value of type 'string' for element 1 of null, but got 2 (int)");
  }

  @Test
  public void testLabelListSyntaxError() throws Exception {
    Object list = Arrays.<Object>asList("//foo:bar/..", "foo");
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> BuildType.LABEL_LIST.convert(list, "myexpr", currentRule));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "invalid label '//foo:bar/..' in element 0 of myexpr: "
                + "invalid target name 'bar/..': "
                + "target names may not contain up-level references '..'");
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
    assertThat(converted).isNotSameInstanceAs(expected);
    assertThat(collectLabels(Type.STRING_LIST_DICT, converted)).isEmpty();
  }

  @Test
  public void testStringListDictBadFirstElement() throws Exception {
    Object input =
        ImmutableMap.of(
            StarlarkInt.of(2), Arrays.asList("foo", "bar"), "wiz", Arrays.asList("bang"));
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> Type.STRING_LIST_DICT.convert(input, null, currentRule));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("expected value of type 'string' for dict key element, but got 2 (int)");
  }

  @Test
  public void testStringListDictBadSecondElement() throws Exception {
    Object input = ImmutableMap.of("foo", "bar",
                                   "wiz", Arrays.asList("bang"));
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class,
            () -> Type.STRING_LIST_DICT.convert(input, null, currentRule));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "expected value of type 'list(string)' for dict value element, "
                + "but got \"bar\" (string)");
  }

  @Test
  public void testStringListDictBadElements1() throws Exception {
    Object input = ImmutableMap.of(Tuple.of("foo"), Tuple.of("bang"), "wiz", Tuple.of("bang"));
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class, () -> Type.STRING_LIST_DICT.convert(input, null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "expected value of type 'string' for dict key element, but got "
                + "(\"foo\",) (tuple)");
  }

  @Test
  public void testStringDictThrowsConversionException() throws Exception {
    Type.ConversionException e =
        assertThrows(
            Type.ConversionException.class, () -> Type.STRING_DICT.convert("some string", null));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "expected value of type 'dict(string, string)', but got \"some string\" (string)");
  }

  private static ImmutableList<Label> collectLabels(Type<?> type, Object value) {
    final ImmutableList.Builder<Label> result = ImmutableList.builder();
    type.visitLabels((label, dummy) -> result.add(label), value, /*context=*/ null);
    return result.build();
  }
}
