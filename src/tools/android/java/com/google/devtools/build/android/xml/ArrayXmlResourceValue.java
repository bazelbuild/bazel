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
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;
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
 * <li>Typed arrays (http://developer.android.com/guide/topics/resources/more-resources
 * .html#TypedArray) which which are indicated by a &lt;array&gt; tag.</li>
 * <li>Integer array (http://developer.android.com/guide/topics/resources/more-resources
 * .html#IntegerArray) which are indicated by &lt;integer-array&gt; tag.</li>
 * <li>String array (http://developer.android.com/guide/topics/resources/string-resource
 * .html#StringArray) which are indicated by &lt;string-array&gt; tag.</li>
 * </ul>
 *
 * Both of these are accessed by R.array.&lt;name&gt; in java.
 */
@Immutable
public class ArrayXmlResourceValue implements XmlResourceValue {
  private static final QName TAG_INTEGER_ARRAY = QName.valueOf("integer-array");
  private static final QName TAG_ARRAY = QName.valueOf("array");
  private static final QName TAG_STRING_ARRAY = QName.valueOf("string-array");
  private static final Function<String, String> ITEM_TO_XML =
      new Function<String, String>() {
        @Nullable
        @Override
        public String apply(@Nullable String item) {
          return String.format("<item>%s</item>", item);
        }
      };

  /**
   * Enumerates the different types of array tags.
   */
  public enum ArrayType {
    INTEGER_ARRAY(TAG_INTEGER_ARRAY),
    ARRAY(TAG_ARRAY),
    STRING_ARRAY(TAG_STRING_ARRAY);

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

    String openTag(FullyQualifiedName key) {
      return String.format("<%s name='%s'>", tagName.getLocalPart(), key.name());
    }

    String closeTag() {
      return String.format("</%s>", tagName.getLocalPart());
    }
  }

  private final ImmutableList<String> values;
  private final ArrayType arrayType;

  private ArrayXmlResourceValue(ArrayType arrayType, ImmutableList<String> values) {
    this.arrayType = arrayType;
    this.values = values;
  }

  @VisibleForTesting
  public static XmlResourceValue of(ArrayType arrayType, String... values) {
    return of(arrayType, Arrays.asList(values));
  }

  public static XmlResourceValue of(ArrayType arrayType, List<String> values) {
    return new ArrayXmlResourceValue(arrayType, ImmutableList.copyOf(values));
  }

  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(ArrayType.valueOf(proto.getValueType()), proto.getListValueList());
  }

  @Override
  public void write(
      FullyQualifiedName key, Path source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.writeToValuesXml(
        key,
        FluentIterable.from(
                ImmutableList.of(String.format("<!-- %s -->", source), arrayType.openTag(key)))
            .append(FluentIterable.from(values).transform(ITEM_TO_XML))
            .append(arrayType.closeTag()));
  }

  @Override
  public int serializeTo(Path source, OutputStream output) throws IOException {
    return XmlResourceValues.serializeProtoDataValue(
        output,
        XmlResourceValues.newSerializableDataValueBuilder(source)
            .setXmlValue(
                SerializeFormat.DataValueXml.newBuilder()
                    .addAllListValue(values)
                    .setType(SerializeFormat.DataValueXml.XmlType.ARRAY)
                    .setValueType(arrayType.toString())));
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

  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    throw new IllegalArgumentException(this + " is not a combinable resource.");
  }

  public static XmlResourceValue parseArray(XMLEventReader eventReader, StartElement start)
      throws XMLStreamException {
    List<String> values = new ArrayList<>();
    for (XMLEvent element = XmlResourceValues.nextTag(eventReader);
        !XmlResourceValues.isEndTag(element, start.getName());
        element = XmlResourceValues.nextTag(eventReader)) {
      if (XmlResourceValues.isItem(element)) {
        if (!element.isStartElement()) {
          throw new XMLStreamException(
              String.format("Expected start element %s", element), element.getLocation());
        }
        String contents = XmlResourceValues.readContentsAsString(eventReader,
            element.asStartElement().getName());
        values.add(contents != null ? contents : "");
      }
    }
    try {
      return of(ArrayType.fromTagName(start), values);
    } catch (IllegalArgumentException e) {
      throw new XMLStreamException(e.getMessage(), start.getLocation());
    }
  }
}
