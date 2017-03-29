// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidResourceClassWriter;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.Builder;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import java.util.Objects;

/**
 * An attribute associated with the root &lt;resources&gt; tag of an xml file in a values folder.
 */
public class ResourcesAttribute implements XmlResourceValue {

  public static ResourcesAttribute of(String name, String value) {
    return new ResourcesAttribute(name, value);
  }

  public static ResourcesAttribute from(SerializeFormat.DataValueXml proto) {
    Map.Entry<String, String> attribute =
        Iterables.getOnlyElement(proto.getAttributeMap().entrySet());
    return of(attribute.getKey(), attribute.getValue());
  }

  private final String name;
  private final String value;

  private ResourcesAttribute(String name, String value) {
    this.name = name;
    this.value = value;
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.defineAttribute(key, value);
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    SerializeFormat.DataValue.Builder builder =
        XmlResourceValues.newSerializableDataValueBuilder(sourceId);
    Builder xmlValueBuilder =
        builder
            .getXmlValueBuilder()
            .putAllNamespace(namespaces.asMap())
            .setType(SerializeFormat.DataValueXml.XmlType.RESOURCES_ATTRIBUTE)
            .putAttribute(name, value);
    builder.setXmlValue(xmlValueBuilder);
    return XmlResourceValues.serializeProtoDataValue(output, builder);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, value);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof ResourcesAttribute)) {
      return false;
    }
    ResourcesAttribute other = (ResourcesAttribute) obj;
    return Objects.equals(name, other.name)
        && Objects.equals(value, other.value);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("name", name)
        .add("value", value)
        .toString();
  }

  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    if (!(value instanceof ResourcesAttribute)) {
      throw new IllegalArgumentException(value + "is not combinable with " + this);
    }
    ResourcesAttribute other = (ResourcesAttribute) value;
    if ((name.startsWith("tools:keep") && other.name.startsWith("tools:keep"))
        || (name.startsWith("tools:discard") && other.name.startsWith("tools:discard"))) {
      return of(name, Joiner.on(',').join(value, other.value));
    }
    throw new IllegalArgumentException(value + "is not combinable with " + this);
  }

  @Override
  public void writeResourceToClass(
      FullyQualifiedName key, AndroidResourceClassWriter resourceClassWriter) {
    // This is an xml attribute and does not have any java representation.
  }

  @Override
  public String asConflictStringWith(DataSource source) {
    if (value != null) {
      return String.format(" %s (with value %s)", source.asConflictString(), value);
    }
    return source.asConflictString();
  }
}
