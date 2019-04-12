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

import com.android.aapt.Resources.Plural;
import com.android.aapt.Resources.Value;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableMap;
import com.google.common.xml.XmlEscapers;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidDataWritingVisitor.ValuesResourceDefinition;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.concurrent.Immutable;
import javax.xml.namespace.QName;

/**
 * Represents an Android Plural Resource.
 *
 * <p>
 * Plurals are a localization construct (http://developer.android.com/guide/topics/resources/
 * string-resource.html#Plurals) that are basically a map of key to value. They are defined in xml
 * as: <code>
 *   &lt;plurals name="plural_name"&gt;
 *     &lt;item quantity=["zero" | "one" | "two" | "few" | "many" | "other"]&gt;
 *       text_string
 *     &lt;/item&gt;
 *   &lt;/plurals&gt;
 * </code>
 */
@Immutable
public class PluralXmlResourceValue implements XmlResourceValue {

  private static final QName PLURALS = QName.valueOf("plurals");

  private final ImmutableMap<String, String> values;

  private final ImmutableMap<String, String> attributes;

  private PluralXmlResourceValue(
      ImmutableMap<String, String> attributes, ImmutableMap<String, String> values) {
    this.attributes = attributes;
    this.values = values;
  }

  public static XmlResourceValue createWithoutAttributes(ImmutableMap<String, String> values) {
    return createWithAttributesAndValues(ImmutableMap.<String, String>of(), values);
  }

  public static XmlResourceValue createWithAttributesAndValues(
      ImmutableMap<String, String> attributes, ImmutableMap<String, String> values) {
    return new PluralXmlResourceValue(attributes, values);
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {

    ValuesResourceDefinition definition =
        mergedDataWriter
            .define(key)
            .derivedFrom(source)
            .startTag(PLURALS)
            .named(key)
            .addAttributesFrom(attributes.entrySet())
            .closeTag();

    for (Map.Entry<String, String> plural : values.entrySet()) {
      definition =
          definition
              .startItemTag()
              .attribute("quantity")
              .setTo(plural.getKey())
              .closeTag()
              .addCharactersOf(plural.getValue())
              .endTag()
              .addCharactersOf("\n");
    }
    definition.endTag().save();
  }

  @Override
  public void writeResourceToClass(FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(key.type(), key.name());
  }

  @Override
  public int hashCode() {
    return Objects.hash(attributes, values);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof PluralXmlResourceValue)) {
      return false;
    }
    PluralXmlResourceValue other = (PluralXmlResourceValue) obj;
    return Objects.equals(values, other.values) && Objects.equals(attributes, other.attributes);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("values", values)
        .add("attributes", attributes)
        .toString();
  }

  @SuppressWarnings("deprecation")
  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return createWithAttributesAndValues(
        ImmutableMap.copyOf(proto.getAttribute()),
        ImmutableMap.copyOf(proto.getMappedStringValue()));
  }

  public static XmlResourceValue from(Value proto) {
    Plural plural = proto.getCompoundValue().getPlural();

    Map<String, String> items = new HashMap<>();

    for (Plural.Entry entry : plural.getEntryList()) {
      String name = entry.getArity().toString().toLowerCase();
      String value =
          XmlEscapers.xmlContentEscaper().escape(
              entry.getItem()
                  .getStr()
                  .getValue());
      items.put(name, value);
    }

    return createWithAttributesAndValues(
        ImmutableMap.of(),
        ImmutableMap.copyOf(items));
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    SerializeFormat.DataValue.Builder builder =
        XmlResourceValues.newSerializableDataValueBuilder(sourceId);
    SerializeFormat.DataValue value =
        builder
            .setXmlValue(
                builder
                    .getXmlValueBuilder()
                    .setType(XmlType.PLURAL)
                    .putAllNamespace(namespaces.asMap())
                    .putAllAttribute(attributes)
                    .putAllMappedStringValue(values))
            .build();
    value.writeDelimitedTo(output);
    return CodedOutputStream.computeUInt32SizeNoTag(value.getSerializedSize())
        + value.getSerializedSize();
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
  public String asConflictStringWith(DataSource source) {
    return source.asConflictString();
  }
}
