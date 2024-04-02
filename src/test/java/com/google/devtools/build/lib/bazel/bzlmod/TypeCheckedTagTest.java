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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Attribute.AllowedValueSet;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Types;
import com.google.devtools.build.lib.util.FileTypeSet;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.Structure;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TypeCheckedTag}. */
@RunWith(JUnit4.class)
public class TypeCheckedTagTest {

  private static Object getattr(Structure structure, String fieldName) throws Exception {
    return Starlark.getattr(
        Mutability.IMMUTABLE,
        StarlarkSemantics.DEFAULT,
        structure,
        fieldName,
        /*defaultValue=*/ null);
  }

  @Test
  public void basic() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(attr("foo", Type.INTEGER).build()),
            buildTag("tag_name").addAttr("foo", StarlarkInt.of(3)).setDevDependency().build(),
            /* labelConverter= */ null);
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo");
    assertThat(getattr(typeCheckedTag, "foo")).isEqualTo(StarlarkInt.of(3));
    assertThat(typeCheckedTag.isDevDependency()).isTrue();
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
            new LabelConverter(
                PackageIdentifier.parse("@myrepo//mypkg"),
                createRepositoryMapping(createModuleKey("test", "1.0"), "repo", "other_repo")));
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo");
    assertThat(getattr(typeCheckedTag, "foo"))
        .isEqualTo(
            StarlarkList.immutableOf(
                Label.parseCanonicalUnchecked("@myrepo//mypkg:thing1"),
                Label.parseCanonicalUnchecked("@myrepo//pkg:thing2"),
                Label.parseCanonicalUnchecked("@other_repo//pkg:thing3")));
    assertThat(typeCheckedTag.isDevDependency()).isFalse();
  }

  @Test
  public void label_withoutDefaultValue() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(
                attr("foo", BuildType.LABEL).allowedFileTypes(FileTypeSet.ANY_FILE).build()),
            buildTag("tag_name").setDevDependency().build(),
            new LabelConverter(
                PackageIdentifier.parse("@myrepo//mypkg"),
                createRepositoryMapping(createModuleKey("test", "1.0"), "repo", "other_repo")));
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo");
    assertThat(getattr(typeCheckedTag, "foo")).isEqualTo(Starlark.NONE);
    assertThat(typeCheckedTag.isDevDependency()).isTrue();
  }

  @Test
  public void stringListDict_default() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(
                attr("foo", Types.STRING_LIST_DICT)
                    .value(ImmutableMap.of("key", ImmutableList.of("value1", "value2")))
                    .build()),
            buildTag("tag_name").build(),
            null);
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo");
    assertThat(getattr(typeCheckedTag, "foo"))
        .isEqualTo(
            Dict.builder()
                .put("key", StarlarkList.immutableOf("value1", "value2"))
                .buildImmutable());
    assertThat(typeCheckedTag.isDevDependency()).isFalse();
  }

  @Test
  public void multipleAttributesAndDefaults() throws Exception {
    TypeCheckedTag typeCheckedTag =
        TypeCheckedTag.create(
            createTagClass(
                attr("foo", Type.STRING).mandatory().build(),
                attr("bar", Type.INTEGER).value(StarlarkInt.of(3)).build(),
                attr("quux", Types.STRING_LIST).build()),
            buildTag("tag_name")
                .addAttr("foo", "fooValue")
                .addAttr("quux", StarlarkList.immutableOf("quuxValue1", "quuxValue2"))
                .build(),
            /* labelConverter= */ null);
    assertThat(typeCheckedTag.getFieldNames()).containsExactly("foo", "bar", "quux");
    assertThat(getattr(typeCheckedTag, "foo")).isEqualTo("fooValue");
    assertThat(getattr(typeCheckedTag, "bar")).isEqualTo(StarlarkInt.of(3));
    assertThat(getattr(typeCheckedTag, "quux"))
        .isEqualTo(StarlarkList.immutableOf("quuxValue1", "quuxValue2"));
    assertThat(typeCheckedTag.isDevDependency()).isFalse();
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
                    /*labelConverter=*/ null));
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
                    /*labelConverter=*/ null));
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
                    /*labelConverter=*/ null));
    assertThat(e).hasMessageThat().contains("unknown attribute bar provided");
  }
}
