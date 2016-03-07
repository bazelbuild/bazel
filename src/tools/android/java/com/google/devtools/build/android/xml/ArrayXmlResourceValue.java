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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;

import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import javax.xml.namespace.QName;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/**
 * Represents an Android resource array.
 *
 * There are two flavors of Android Resource arrays:
 * <ul>
 *   <li>Typed arrays (http://developer.android.com/guide/topics/resources/more-resources
 *   .html#TypedArray) which which are indicated by a &lt;array&gt; tag.</li>
 *   <li>Integer array (http://developer.android.com/guide/topics/resources/more-resources
 *   .html#IntegerArray) which are indicated by &lt;integer-array&gt; tag.</li>
 * </ul>
 *
 * Both of these are accessed by R.array.&lt;name&gt; in java.
 */
public class ArrayXmlResourceValue implements XmlResourceValue {
  private static final QName TAG_INTEGER_ARRAY = QName.valueOf("integer-array");
  private static final QName TAG_ARRAY = QName.valueOf("array");

  /**
   * Enumerates the different types of array tags.
   */
  public enum ArrayType {
    INTEGER_ARRAY(TAG_INTEGER_ARRAY),
    ARRAY(TAG_ARRAY);

    public QName tagName;

    ArrayType(QName tagName) {
      this.tagName = tagName;
    }

    public static ArrayType fromTagName(StartElement start) {
      final QName tagQName = start.getName();
      for (ArrayType arrayType : values()) {
        if (arrayType.tagName.equals(tagQName)) {
          return arrayType;
        }
      }
      throw new IllegalArgumentException(
          String.format("%s not found in %s", tagQName, Arrays.toString(values())));
    }
  }

  private final List<String> values;
  private ArrayType arrayType;

  private ArrayXmlResourceValue(ArrayType arrayType, List<String> values) {
    this.arrayType = arrayType;
    this.values = values;
  }

  @VisibleForTesting
  public static XmlResourceValue of(ArrayType arrayType, String... values) {
    return of(arrayType, Arrays.asList(values));
  }

  public static XmlResourceValue of(ArrayType arrayType, List<String> values) {
    return new ArrayXmlResourceValue(arrayType, values);
  }

  @Override
  public void write(Writer buffer, FullyQualifiedName name) {
    // TODO(corysmith): Implement writing xml.
    throw new UnsupportedOperationException();
  }

  @Override
  public int hashCode() {
    return Objects.hash(arrayType, values);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof ArrayXmlResourceValue)) {
      return false;
    }
    ArrayXmlResourceValue other = (ArrayXmlResourceValue) obj;
    return Objects.equals(arrayType, other.arrayType) && Objects.equals(values, other.values);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("arrayType", arrayType)
        .add("values", values)
        .toString();
  }

  public static XmlResourceValue parseArray(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    List<String> values = new ArrayList<>();
    for (XMLEvent element = eventReader.nextTag();
        !XmlResourceValues.isEndTag(element, start.getName());
        element = eventReader.nextTag()) {
      if (XmlResourceValues.isItem(element)) {
        values.add(eventReader.getElementText());
      }
    }
    return of(ArrayType.fromTagName(start), values);
  }
}
