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

import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;
import com.google.protobuf.CodedOutputStream;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Map.Entry;
import java.util.Objects;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

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

  public static final Function<Entry<String, String>, String> ENTRY_TO_PLURAL =
      new Function<Entry<String, String>, String>() {
        @Nullable
        @Override
        public String apply(Entry<String, String> input) {
          return String.format("<item quantity='%s'>%s</item>", input.getKey(), input.getValue());
        }
      };
  private final ImmutableMap<String, String> values;

  private PluralXmlResourceValue(ImmutableMap<String, String> values) {
    this.values = values;
  }

  public static XmlResourceValue of(ImmutableMap<String, String> values) {
    return new PluralXmlResourceValue(values);
  }

  @Override
  public void write(
      FullyQualifiedName key, Path source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.writeToValuesXml(
        key,
        FluentIterable.from(
                ImmutableList.of(
                    String.format("<!-- %s -->", source),
                    String.format("<plurals name='%s'>", key.name())))
            .append(FluentIterable.from(values.entrySet()).transform(ENTRY_TO_PLURAL))
            .append("</plurals>"));
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

  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(ImmutableMap.copyOf(proto.getMappedStringValue()));
  }

  @Override
  public int serializeTo(Path source, OutputStream output) throws IOException {
    SerializeFormat.DataValue.Builder builder =
        XmlResourceValues.newSerializableDataValueBuilder(source);
    SerializeFormat.DataValue value =
        builder
            .setXmlValue(
                builder
                    .getXmlValueBuilder()
                    .setType(XmlType.PLURAL)
                    .putAllMappedStringValue(values))
            .build();
    value.writeDelimitedTo(output);
    return CodedOutputStream.computeUInt32SizeNoTag(value.getSerializedSize())
        + value.getSerializedSize();
  }
}
