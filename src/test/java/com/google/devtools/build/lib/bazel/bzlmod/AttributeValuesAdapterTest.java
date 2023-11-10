// Copyright 2023 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.AttributeValuesAdapter.STRING_ESCAPE_SEQUENCE;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.util.Map;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.Tuple;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class AttributeValuesAdapterTest extends FoundationTestCase {

  @Test
  public void testAttributeValuesAdapter() throws IOException {
    Dict.Builder<String, Object> dict = new Dict.Builder<>();
    Label l1 = Label.parseCanonicalUnchecked("@//foo:bar");
    Label l2 = Label.parseCanonicalUnchecked("@//foo:tar");
    dict.put("Integer", StarlarkInt.of(56));
    dict.put("Boolean", false);
    dict.put("String", "Hello String");
    dict.put("StringWithAngleBrackets", "<Hello>");
    dict.put(
        "LabelLikeString",
        StarlarkList.of(Mutability.IMMUTABLE, "@//foo:bar", ":baz", "@@//baz:quz"));
    dict.put(
        "StringsWithEscapeSequence",
        StarlarkList.of(
            Mutability.IMMUTABLE,
            "@@//foo:bar" + STRING_ESCAPE_SEQUENCE,
            STRING_ESCAPE_SEQUENCE + "@@//foo:bar",
            STRING_ESCAPE_SEQUENCE + "@@//foo:bar" + STRING_ESCAPE_SEQUENCE,
            STRING_ESCAPE_SEQUENCE
                + STRING_ESCAPE_SEQUENCE
                + "@@//foo:bar"
                + STRING_ESCAPE_SEQUENCE
                + STRING_ESCAPE_SEQUENCE));
    dict.put("Label", l1);
    dict.put(
        "ListOfInts", StarlarkList.of(Mutability.IMMUTABLE, StarlarkInt.of(1), StarlarkInt.of(2)));
    dict.put("ListOfLabels", StarlarkList.of(Mutability.IMMUTABLE, l1, l2));
    dict.put("ListOfStrings", StarlarkList.of(Mutability.IMMUTABLE, "Hello", "There!"));
    Dict.Builder<Label, String> dictLabelString = new Dict.Builder<>();
    dictLabelString.put(l1, "Label#1");
    dictLabelString.put(l2, "Label#2");
    dict.put("DictOfLabel-String", dictLabelString.buildImmutable());

    Dict<String, Object> builtDict = dict.buildImmutable();
    AttributeValuesAdapter attrAdapter = new AttributeValuesAdapter();
    String jsonString;
    try (StringWriter stringWriter = new StringWriter()) {
      attrAdapter.write(new JsonWriter(stringWriter), AttributeValues.create(builtDict));
      jsonString = stringWriter.toString();
    }
    AttributeValues attributeValues;
    try (StringReader stringReader = new StringReader(jsonString)) {
      attributeValues = attrAdapter.read(new JsonReader(stringReader));
    }

    // Verify that the JSON string does not contain any escaped angle brackets.
    assertThat(jsonString).doesNotContain("\\u003c");
    // Verify that the String "Hello String" is preserved as is, without any additional escaping.
    assertThat(jsonString).contains(":\"Hello String\"");
    assertThat((Map<?, ?>) attributeValues.attributes()).containsExactlyEntriesIn(builtDict);
  }

  @Test
  public void testTuple() throws IOException {
    Dict.Builder<String, Object> dict = new Dict.Builder<>();
    dict.put("Tuple", Tuple.of("bzl", "mod"));

    Dict<String, Object> builtDict = dict.buildImmutable();
    AttributeValuesAdapter attrAdapter = new AttributeValuesAdapter();
    String jsonString;
    try (StringWriter stringWriter = new StringWriter()) {
      attrAdapter.write(new JsonWriter(stringWriter), AttributeValues.create(builtDict));
      jsonString = stringWriter.toString();
    }
    AttributeValues attributeValues;
    try (StringReader stringReader = new StringReader(jsonString)) {
      attributeValues = attrAdapter.read(new JsonReader(stringReader));
    }
    assertThat((Map<?, ?>) attributeValues.attributes())
        .containsExactly("Tuple", StarlarkList.of(Mutability.IMMUTABLE, "bzl", "mod"));
  }
}
