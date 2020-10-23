// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import com.android.aapt.Resources.Item;
import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.Value;
import com.google.devtools.build.android.resources.Visibility;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link IdXmlResourceValue}. */
@RunWith(JUnit4.class)
public final class IdXmlResourceValueTest {

  @Test
  public void combineWith_conflict() {
    // Note that we're testing both the IdXmlResourceValue constructor as well as "combineWith";
    // there are no accessors for its internals and it's not worth adding any for testing.
    IdXmlResourceValue alias1 =
        IdXmlResourceValue.from(
            Value.newBuilder()
                .setItem(Item.newBuilder().setRef(Reference.newBuilder().setName("@id/alias1")))
                .build(),
            Visibility.UNKNOWN);
    IdXmlResourceValue alias2 =
        IdXmlResourceValue.from(
            Value.newBuilder()
                .setItem(Item.newBuilder().setRef(Reference.newBuilder().setName("@id/alias2")))
                .build(),
            Visibility.UNKNOWN);

    assertThat(alias1).isNotEqualTo(alias2);
    assertThrows(IllegalArgumentException.class, () -> alias1.combineWith(alias2));
    assertThrows(IllegalArgumentException.class, () -> alias2.combineWith(alias1));
  }

  @Test
  public void combineWith_specifiedRefOverwritesEmpty() {
    IdXmlResourceValue hasRef =
        IdXmlResourceValue.from(
            Value.newBuilder()
                .setItem(Item.newBuilder().setRef(Reference.newBuilder().setName("@id/alias1")))
                .build(),
            Visibility.UNKNOWN);
    IdXmlResourceValue empty =
        IdXmlResourceValue.from(Value.getDefaultInstance(), Visibility.UNKNOWN);

    assertThat(empty.combineWith(hasRef)).isEqualTo(hasRef);
    assertThat(hasRef.combineWith(empty)).isEqualTo(hasRef);
  }

  @Test
  public void combineWith_mergeVisibility() {
    IdXmlResourceValue pub = IdXmlResourceValue.from(Value.getDefaultInstance(), Visibility.PUBLIC);
    IdXmlResourceValue priv =
        IdXmlResourceValue.from(Value.getDefaultInstance(), Visibility.PRIVATE);

    assertThat(pub.combineWith(priv)).isEqualTo(pub);
    assertThat(priv.combineWith(pub)).isEqualTo(pub);
  }
}
