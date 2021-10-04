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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.buildTag;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createRepositoryMapping;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createTagClass;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuildType.LabelConversionContext;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.HashMap;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TypeCheckedTag}. */
@RunWith(JUnit4.class)
public class TypeCheckedTagTest {

  @Test
  public void basic() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(attr("foo", Type.INTEGER).build()),
            buildTag("tag_name").addAttr("foo", StarlarkInt.of(3)).build(),
            /*labelConversionContext=*/ null);
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo");
    assertThat(typeCheckedTag.getValue("foo")).isEqualTo(StarlarkInt.of(3));
  }

  @Test
  public void label() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(
                attr("foo", BuildType.LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE).build()),
            buildTag("tag_name")
                .addAttr(
                    "foo", StarlarkList.immutableOf(":thing1", "//pkg:thing2", "@repo//pkg:thing3"))
                .build(),
            new LabelConversionContext(
                Label.parseAbsoluteUnchecked("@myrepo//mypkg:defs.bzl"),
                createRepositoryMapping(createModuleKey("test", "1.0"), "repo", "other_repo"),
                new HashMap<>()));
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo");
    assertThat(typeCheckedTag.getValue("foo"))
        .isEqualTo(
            StarlarkList.immutableOf(
                Label.parseAbsoluteUnchecked("@myrepo//mypkg:thing1"),
                Label.parseAbsoluteUnchecked("@myrepo//pkg:thing2"),
                Label.parseAbsoluteUnchecked("@other_repo//pkg:thing3")));
  }

  @Test
  public void multipleAttributesAndDefaults() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(
                attr("foo", Type.STRING).mandatory().build(),
                attr("bar", Type.INTEGER).value(StarlarkInt.of(3)).build(),
                attr("quux", Type.STRING_LIST).build()),
            buildTag("tag_name")
                .addAttr("foo", "fooValue")
                .addAttr("quux", StarlarkList.immutableOf("quuxValue1", "quuxValue2"))
                .build(),
            /*labelConversionContext=*/ null);
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo", "bar", "quux");
    assertThat(typeCheckedTag.getValue("foo")).isEqualTo("fooValue");
    assertThat(typeCheckedTag.getValue("bar")).isEqualTo(StarlarkInt.of(3));
    assertThat(typeCheckedTag.getValue("quux"))
        .isEqualTo(StarlarkList.immutableOf("quuxValue1", "quuxValue2"));
  }

  @Test
  public void mandatory() throws Exception {
    ExternalDepsException e =
        assertThrows(
            ExternalDepsException.class,
            () ->
                TypeCheckedTag.create(
                    createTagClass(attr("foo", Type.STRING).mandatory().build()),
                    buildTag("tag_name").build(),
                    /*labelConversionContext=*/ null));
    assertThat(e).hasMessageThat().contains("mandatory attribute foo isn't being specified");
  }

  @Test
  public void allowedValues() throws Exception {
    ExternalDepsException e =
        assertThrows(
            ExternalDepsException.class,
            () ->
                TypeCheckedTag.create(
                    createTagClass(
                        attr("foo", Type.STRING)
                            .allowedValues(new AllowedValueSet("yes", "no"))
                            .build()),
                    buildTag("tag_name").addAttr("foo", "maybe").build(),
                    /*labelConversionContext=*/ null));
    assertThat(e)
        .hasMessageThat()
        .contains("the value for attribute foo has to be one of 'yes' or 'no' instead of 'maybe'");
  }

  @Test
  public void unknownAttr() throws Exception {
    ExternalDepsException e =
        assertThrows(
            ExternalDepsException.class,
            () ->
                TypeCheckedTag.create(
                    createTagClass(attr("foo", Type.STRING).build()),
                    buildTag("tag_name").addAttr("bar", "maybe").build(),
                    /*labelConversionContext=*/ null));
    assertThat(e).hasMessageThat().contains("unknown attribute bar provided");
  }
}
