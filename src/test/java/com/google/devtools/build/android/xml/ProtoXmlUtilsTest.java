// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.xml;

import static com.google.common.truth.Truth.assertThat;

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.XmlAttribute;
import com.android.aapt.Resources.XmlElement;
import com.android.aapt.Resources.XmlNode;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ProtoXmlUtils}. */
@RunWith(JUnit4.class)
public final class ProtoXmlUtilsTest {

  @Test
  public void parseAttributeNameReference() {
    assertThat(
            ProtoXmlUtils.parseAttributeNameReference(
                "http://schemas.android.com/apk/res-auto", "foo"))
        .isEqualTo(Optional.of(Reference.newBuilder().setName("attr/foo").build()));
    assertThat(
            ProtoXmlUtils.parseAttributeNameReference(
                "http://schemas.android.com/apk/res/android", "foo"))
        .isEqualTo(Optional.of(Reference.newBuilder().setName("android:attr/foo").build()));
    assertThat(
            ProtoXmlUtils.parseAttributeNameReference(
                "http://schemas.android.com/apk/prv/res/android", "foo"))
        .isEqualTo(
            Optional.of(
                Reference.newBuilder().setPrivate(true).setName("android:attr/foo").build()));

    assertThat(ProtoXmlUtils.parseAttributeNameReference("", "foo")).isEqualTo(Optional.empty());
    assertThat(ProtoXmlUtils.parseAttributeNameReference("http://asdf", "foo"))
        .isEqualTo(Optional.empty());
  }

  @Test
  public void parseResourceReference() {
    assertThat(ProtoXmlUtils.parseResourceReference("@string/foo"))
        .isEqualTo(
            Optional.of(
                Reference.newBuilder()
                    .setType(Reference.Type.REFERENCE)
                    .setName("string/foo")
                    .build()));
    assertThat(ProtoXmlUtils.parseResourceReference("?*android:attr/foo"))
        .isEqualTo(
            Optional.of(
                Reference.newBuilder()
                    .setType(Reference.Type.ATTRIBUTE)
                    .setPrivate(true)
                    .setName("android:attr/foo")
                    .build()));

    assertThat(ProtoXmlUtils.parseResourceReference("x")).isEqualTo(Optional.empty());
    assertThat(ProtoXmlUtils.parseResourceReference("@")).isEqualTo(Optional.empty());
    assertThat(ProtoXmlUtils.parseResourceReference("@x")).isEqualTo(Optional.empty());
    assertThat(ProtoXmlUtils.parseResourceReference("@x/foo")).isEqualTo(Optional.empty());
  }

  @Test
  public void getAllResourceReferences() {
    XmlNode root =
        XmlNode.newBuilder()
            .setElement(
                XmlElement.newBuilder()
                    .addAttribute(
                        XmlAttribute.newBuilder()
                            .setNamespaceUri("http://schemas.android.com/apk/res/android")
                            .setName("text")
                            .setValue("asdf"))
                    .addChild(
                        XmlNode.newBuilder()
                            .setElement(
                                XmlElement.newBuilder()
                                    .addAttribute(
                                        XmlAttribute.newBuilder()
                                            .setName("id")
                                            .setValue("@string/foo")))))
            .build();

    assertThat(ProtoXmlUtils.getAllResourceReferences(root))
        .containsExactly(
            Reference.newBuilder().setName("android:attr/text").build(),
            Reference.newBuilder().setName("string/foo").build());
  }
}
