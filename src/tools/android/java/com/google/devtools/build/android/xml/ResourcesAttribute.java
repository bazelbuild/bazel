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

import com.android.aapt.Resources.Reference;
import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.DependencyInfo;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml;
import com.google.devtools.build.android.resources.Visibility;
import com.google.errorprone.annotations.Immutable;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Map;
import java.util.Objects;

/**
 * An attribute associated with the root &lt;resources&gt; tag of an xml file in a values folder.
 */
public class ResourcesAttribute implements XmlResourceValue {

  @Immutable
  private interface Combiner {
    public String combine(String first, String second);
  }

  private static final Combiner COMMA_SEPARATED_COMBINER = new Combiner() {
    private final Joiner joiner = Joiner.on(',');
    @Override
    public String combine(String first, String second) {
      return joiner.join(first, second);
    }
  };

  /**
   * Represents the semantic meaning of an xml attribute and how it is combined with other
   * attributes of the same type.
   */
  public static enum AttributeType {
    UNCOMBINABLE(null, null),
    TOOLS_IGNORE("{http://schemas.android.com/tools}ignore", COMMA_SEPARATED_COMBINER),
    TOOLS_MENU("{http://schemas.android.com/tools}menu", COMMA_SEPARATED_COMBINER),
    TOOLS_KEEP("{http://schemas.android.com/tools}keep", COMMA_SEPARATED_COMBINER),
    TOOLS_DISCARD("{http://schemas.android.com/tools}discard", COMMA_SEPARATED_COMBINER);

    public static AttributeType from(String name) {
      if (name == null) {
        return UNCOMBINABLE;
      }
      for (AttributeType type : AttributeType.values()) {
        if (name.equals(type.name)) {
          return type;
        }
      }
      return UNCOMBINABLE;
    }

    private final String name;
    public final Combiner combiner;

    private AttributeType(String name, Combiner combiner) {
      this.name = name;
      this.combiner = combiner;
    }

    public boolean isCombining() {
      return this != UNCOMBINABLE;
    }
  }

  public static ResourcesAttribute of(FullyQualifiedName fqn, String name, String value) {
    return new ResourcesAttribute(AttributeType.from(fqn.name()), name, value);
  }

  public static ResourcesAttribute from(SerializeFormat.DataValueXml proto) {
    Map.Entry<String, String> attribute =
        Iterables.getOnlyElement(proto.getAttributeMap().entrySet());
    return new ResourcesAttribute(
        AttributeType.valueOf(proto.getValueType()),
        attribute.getKey(),
        attribute.getValue());
  }

  private final AttributeType type;
  private final String name;
  private final String value;

  private ResourcesAttribute(AttributeType type, String name, String value) {
    this.type = type;
    this.name = name;
    this.value = value;
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.defineAttribute(key, name, value);
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    SerializeFormat.DataValue.Builder builder =
        XmlResourceValues.newSerializableDataValueBuilder(sourceId);
    DataValueXml.Builder xmlValueBuilder =
        builder
            .getXmlValueBuilder()
            .putAllNamespace(namespaces.asMap())
            .setType(SerializeFormat.DataValueXml.XmlType.RESOURCES_ATTRIBUTE)
            .setValueType(type.name())
            .putAttribute(name, value);
    builder.setXmlValue(xmlValueBuilder);
    return XmlResourceValues.serializeProtoDataValue(output, builder);
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, name, value);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof ResourcesAttribute)) {
      return false;
    }
    ResourcesAttribute other = (ResourcesAttribute) obj;
    return Objects.equals(type, other.type)
        && Objects.equals(name, other.name)
        && Objects.equals(value, other.value);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("type", type)
        .add("name", name)
        .add("value", value)
        .toString();
  }

  public boolean isCombining() {
    return type.isCombining();
  }

  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    if (!(value instanceof ResourcesAttribute)) {
      throw new IllegalArgumentException(value + "is not combinable with " + this);
    }
    ResourcesAttribute other = (ResourcesAttribute) value;
    if (type == other.type && isCombining()) {
      return new ResourcesAttribute(type, name, type.combiner.combine(this.value, other.value));
    }
    throw new IllegalArgumentException(value + "is not combinable with " + this);
  }

  @Override
  public int compareMergePriorityTo(XmlResourceValue value) {
    return 0;
  }

  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    // This is an xml attribute and does not have any java representation.
  }

  @Override
  public String asConflictStringWith(DataSource source) {
    if (value != null) {
      return String.format(" %s (with value %s)", source.asConflictString(), value);
    }
    return source.asConflictString();
  }

  @Override
  public Visibility getVisibility() {
    return Visibility.UNKNOWN;
  }

  @Override
  public ImmutableList<Reference> getReferencedResources() {
    return ImmutableList.of();
  }
}
