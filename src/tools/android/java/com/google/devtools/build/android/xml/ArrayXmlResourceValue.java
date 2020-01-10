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

import com.android.aapt.Resources.Array;
import com.android.aapt.Resources.Array.Element;
import com.android.aapt.Resources.Item;
import com.android.aapt.Resources.Value;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidDataWritingVisitor.ValuesResourceDefinition;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.DependencyInfo;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.resources.Visibility;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
  /**
   * Enumerates the different types of array parentTags.
   */
  public enum ArrayType {
    INTEGER_ARRAY(TAG_INTEGER_ARRAY),
    ARRAY(TAG_ARRAY),
    STRING_ARRAY(TAG_STRING_ARRAY);

    public final QName tagName;

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

  private final Visibility visibility;
  private final ImmutableList<String> values;
  private final ArrayType arrayType;
  private final ImmutableMap<String, String> attributes;

  private ArrayXmlResourceValue(
      Visibility visibility,
      ArrayType arrayType,
      List<String> values,
      Map<String, String> attributes) {
    this.visibility = visibility;
    this.arrayType = arrayType;
    this.values = ImmutableList.copyOf(values);
    this.attributes = ImmutableMap.copyOf(attributes);
  }

  @VisibleForTesting
  public static XmlResourceValue of(ArrayType arrayType, String... values) {
    return of(arrayType, Arrays.asList(values));
  }

  public static XmlResourceValue of(ArrayType arrayType, List<String> values) {
    return of(arrayType, values, ImmutableMap.<String, String>of());
  }

  public static XmlResourceValue of(
      ArrayType arrayType, List<String> values, ImmutableMap<String, String> attributes) {
    return new ArrayXmlResourceValue(Visibility.UNKNOWN, arrayType, values, attributes);
  }

  @SuppressWarnings("deprecation")
  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(
        ArrayType.valueOf(proto.getValueType()),
        proto.getListValueList(),
        ImmutableMap.copyOf(proto.getAttribute()));
  }

  public static XmlResourceValue from(Value proto, Visibility visibility) {
    Array array = proto.getCompoundValue().getArray();
    List<String> items = new ArrayList<>();

    for (Element entry : array.getElementList()) {
      Item item = entry.getItem();

      if (item.hasPrim()) {
        items.add(SimpleXmlResourceValue.convertPrimitiveToString(item.getPrim()));
      } else if (item.hasRef()) {
        items.add("@" + item.getRef().getName());
      } else if (item.hasStr()) {
        items.add(item.getStr().getValue());
      }
    }

    return new ArrayXmlResourceValue(visibility, ArrayType.ARRAY, items, ImmutableMap.of());
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    ValuesResourceDefinition definition = mergedDataWriter.define(key).derivedFrom(source)
        .startTag(arrayType.tagName)
        .named(key)
        .addAttributesFrom(attributes.entrySet())
        .closeTag();
    for (String value : values) {
      definition =
          definition
              .startItemTag()
              .closeTag()
              .addCharactersOf(value)
              .endTag()
              .addCharactersOf("\n");
    }
    definition.endTag().save();
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    return XmlResourceValues.serializeProtoDataValue(
        output,
        XmlResourceValues.newSerializableDataValueBuilder(sourceId)
            .setXmlValue(
                SerializeFormat.DataValueXml.newBuilder()
                    .addAllListValue(values)
                    .setType(SerializeFormat.DataValueXml.XmlType.ARRAY)
                    .putAllNamespace(namespaces.asMap())
                    .putAllAttribute(attributes)
                    .setValueType(arrayType.toString())));
  }

  @Override
  public int hashCode() {
    return Objects.hash(visibility, arrayType, values, attributes);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof ArrayXmlResourceValue)) {
      return false;
    }
    ArrayXmlResourceValue other = (ArrayXmlResourceValue) obj;
    return Objects.equals(visibility, other.visibility)
        && Objects.equals(arrayType, other.arrayType)
        && Objects.equals(values, other.values)
        && Objects.equals(attributes, other.attributes);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("arrayType", arrayType)
        .add("values", values)
        .add("attributes", attributes)
        .toString();
  }

  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    throw new IllegalArgumentException(this + " is not a combinable resource.");
  }

  @Override
  public int compareMergePriorityTo(XmlResourceValue value) {
    return 0;
  }

  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(dependencyInfo, visibility, key.type(), key.name());
  }

  public static XmlResourceValue parseArray(
      XMLEventReader eventReader, StartElement start, Namespaces.Collector namespacesCollector)
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
        String contents =
            XmlResourceValues.readContentsAsString(
                eventReader, element.asStartElement().getName(), namespacesCollector);
        values.add(contents != null ? contents : "");
      }
    }
    try {
      return of(
          ArrayType.fromTagName(start),
          values,
          ImmutableMap.copyOf(XmlResourceValues.parseTagAttributes(start)));
    } catch (IllegalArgumentException e) {
      throw new XMLStreamException(e.getMessage(), start.getLocation());
    }
  }

  @Override
  public String asConflictStringWith(DataSource source) {
    return source.asConflictString();
  }
}
