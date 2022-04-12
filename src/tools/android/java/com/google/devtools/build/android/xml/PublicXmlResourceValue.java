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

import com.android.SdkConstants;
import com.android.aapt.Resources.Reference;
import com.android.resources.ResourceType;
import com.google.common.base.MoreObjects;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
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
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * Represents an Android resource &lt;public&gt; xml tag.
 *
 * <p>This is used to declare a resource public and reserve a fixed ID for a resource. It is
 * generally undocumented (update this if we ever get a doc), but used heavily by the android
 * framework resources. One description of it is at the android <a
 * href="http://tools.android.com/tech-docs/private-resources">tools site</a>. Public tags can be
 * defined in any xml file in the values folder. <code>
 *   &lt;resources&gt;
 *     &lt;string name="mypublic_string"&gt; Pub &lt;/string&gt;
 *     &lt;public name="mypublic_string" type="string" id="0x7f050004" /&gt;
 *     &lt;string name="myother_string"&gt; the others &lt;/string&gt;
 *     &lt;public name="myother_string" type="string" /&gt;
 *  &lt;/resources&gt;
 * </code> The "id" attribute is optional if an earlier public tag has already specified an "id"
 * attribute. In such cases, ID assignment will continue from the previous reserved ID.
 */
public class PublicXmlResourceValue implements XmlResourceValue {

  private final Map<ResourceType, Optional<Integer>> typeToId;
  private static final String MISSING_ID_VALUE = "";

  private PublicXmlResourceValue(Map<ResourceType, Optional<Integer>> typeToId) {
    this.typeToId = typeToId;
  }

  public static PublicXmlResourceValue of(Map<ResourceType, Optional<Integer>> typeToId) {
    return new PublicXmlResourceValue(typeToId);
  }

  public static XmlResourceValue create(ResourceType type, Optional<Integer> id) {
    Map<ResourceType, Optional<Integer>> map = new EnumMap<>(ResourceType.class);
    map.put(type, id);
    return new PublicXmlResourceValue(map);
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    for (Map.Entry<ResourceType, Optional<Integer>> entry : typeToId.entrySet()) {
      Integer value = entry.getValue().orNull();
      mergedDataWriter
          .define(key)
          .derivedFrom(source)
          .startTag(ResourceType.PUBLIC.getName())
          .named(key)
          .attribute(SdkConstants.ATTR_TYPE)
          .setTo(entry.getKey().toString())
          .optional()
          .attribute(SdkConstants.ATTR_ID)
          .setTo(value == null ? null : "0x" + Integer.toHexString(value))
          .closeUnaryTag()
          .save();
    }
  }

  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {}

  @Override
  public int hashCode() {
    return Objects.hash(typeToId);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof PublicXmlResourceValue)) {
      return false;
    }
    PublicXmlResourceValue other = (PublicXmlResourceValue) obj;
    return Objects.equals(typeToId, other.typeToId);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("typeToId: ", typeToId).toString();
  }

  @SuppressWarnings("deprecation")
  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    Map<String, String> protoValues = proto.getMappedStringValueMap();
    ImmutableMap.Builder<ResourceType, Optional<Integer>> typeToId = ImmutableMap.builder();
    for (Map.Entry<String, String> entry : protoValues.entrySet()) {
      ResourceType type = ResourceType.getEnum(entry.getKey());
      Preconditions.checkNotNull(type);
      Optional<Integer> id =
          MISSING_ID_VALUE.equals(entry.getValue())
              ? Optional.<Integer>absent()
              : Optional.of(Integer.decode(entry.getValue()));
      typeToId.put(type, id);
    }
    return of(typeToId.build());
  }

  public static XmlResourceValue from(ResourceType resourceType, int id) {
    ImmutableMap.Builder<ResourceType, Optional<Integer>> typeToId = ImmutableMap.builder();
    typeToId.put(resourceType, Optional.of(id));
    return of(typeToId.build());
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    Map<String, String> assignments = Maps.newLinkedHashMapWithExpectedSize(typeToId.size());
    for (Map.Entry<ResourceType, Optional<Integer>> entry : typeToId.entrySet()) {
      Optional<Integer> value = entry.getValue();
      String stringValue = value.isPresent() ? value.get().toString() : MISSING_ID_VALUE;
      assignments.put(entry.getKey().toString(), stringValue);
    }
    SerializeFormat.DataValue.Builder builder =
        XmlResourceValues.newSerializableDataValueBuilder(sourceId);
    builder.setXmlValue(
        builder
            .getXmlValueBuilder()
            .setType(SerializeFormat.DataValueXml.XmlType.PUBLIC)
            .putAllNamespace(namespaces.asMap())
            .putAllMappedStringValue(assignments));
    return XmlResourceValues.serializeProtoDataValue(output, builder);
  }

  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    if (!(value instanceof PublicXmlResourceValue)) {
      throw new IllegalArgumentException(value + "is not combinable with " + this);
    }
    PublicXmlResourceValue other = (PublicXmlResourceValue) value;
    Map<ResourceType, Optional<Integer>> combined = new EnumMap<>(ResourceType.class);
    combined.putAll(typeToId);
    for (Map.Entry<ResourceType, Optional<Integer>> entry : other.typeToId.entrySet()) {
      Optional<Integer> existing = combined.get(entry.getKey());
      if (existing != null && !existing.equals(entry.getValue())) {
        throw new IllegalArgumentException(
            String.format(
                "Public resource of type %s assigned two different id values 0x%x and 0x%x",
                entry.getKey(), existing.orNull(), entry.getValue().orNull()));
      }
      combined.put(entry.getKey(), entry.getValue());
    }
    return of(combined);
  }

  @Override
  public int compareMergePriorityTo(XmlResourceValue value) {
    return 0;
  }

  @Override
  public String asConflictStringWith(DataSource source) {
    return source.asConflictString();
  }

  @Override
  public Visibility getVisibility() {
    // <public id="..."> itself is not a value
    return Visibility.UNKNOWN;
  }

  @Override
  public ImmutableList<Reference> getReferencedResources() {
    return ImmutableList.of();
  }
}
