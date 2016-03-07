// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.MoreObjects;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;

import java.io.Writer;
import java.util.Map;
import java.util.Objects;

/**
 * Represents an Android Plural Resource.
 *
 * <p>Plurals are a localization construct (http://developer.android
 * .com/guide/topics/resources/string-resource.html#Plurals) that are basically a map of key to
 * value. They are defined in xml as:
 * <code>
 *   &lt;plurals name="plural_name"&gt;
 *     &lt;item quantity=["zero" | "one" | "two" | "few" | "many" | "other"]&gt;
 *       text_string
 *     &lt;/item&gt;
 *   &lt;/plurals&gt;
 * </code>
 */
public class PluralXmlResourceValue implements XmlResourceValue {

  private Map<String, String> values;

  private PluralXmlResourceValue(Map<String, String> values) {
    this.values = values;
  }

  public static XmlResourceValue of(Map<String, String> values) {
    return new PluralXmlResourceValue(values);
  }

  @Override
  public void write(Writer buffer, FullyQualifiedName name) {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
  }

  @Override
  public int hashCode() {
    return values.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof PluralXmlResourceValue)) {
      return false;
    }
    PluralXmlResourceValue other = (PluralXmlResourceValue) obj;
    return Objects.equals(values, other.values);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("values", values).toString();
  }
}
