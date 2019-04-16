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
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidDataWritingVisitor.StartTag;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Represents an Android Resource id.
 *
 * <p>
 * Ids (http://developer.android.com/guide/topics/resources/more-resources.html#Id) are special --
 * unlike other resources they cannot be overridden. This is due to the nature of their usage. Each
 * id corresponds to context sensitive resource of component, meaning that they have no intrinsic
 * defined value. They exist to reference parts of other resources. Ids can also be declared on the
 * fly in components with the syntax @[+][package:]id/resource_name.
 */
@Immutable
public class IdXmlResourceValue implements XmlResourceValue {

  static final IdXmlResourceValue SINGLETON = new IdXmlResourceValue(null);
  private String value;

  public static XmlResourceValue of() {
    return SINGLETON;
  }
  
  public static XmlResourceValue of(@Nullable String value) {
    if (value == null) {
      return of();
    }
    return new IdXmlResourceValue(value);
  }

  private IdXmlResourceValue(String value) {
    this.value = value;
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    if (!source.isInValuesFolder()) {
      /* Don't write IDs that were never defined in values, into the merged values.xml, to preserve
       * the way initializers are assigned in the R class. Depends on
       * DataSource#combine to accurately determine when a value is ever defined in a
       * values file.
       */
      return;
    }
    StartTag startTag =
        mergedDataWriter
            .define(key)
            .derivedFrom(source)
            .startItemTag()
            .named(key)
            .attribute("type")
            .setTo("id");
    if (value == null) {
      startTag.closeUnaryTag().save();
    } else {
      startTag.closeTag().addCharactersOf(value).endTag().save();
    }
  }

  @Override
  public void writeResourceToClass(FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(key.type(), key.name());
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    DataValueXml.Builder xmlValue =
        SerializeFormat.DataValueXml.newBuilder()
            .setType(XmlType.ID)
            .putAllNamespace(namespaces.asMap());
    if (value != null) {
      xmlValue.setValue(value);
    }
    SerializeFormat.DataValue dataValue =
        XmlResourceValues.newSerializableDataValueBuilder(sourceId).setXmlValue(xmlValue).build();
    dataValue.writeDelimitedTo(output);
    return CodedOutputStream.computeUInt32SizeNoTag(dataValue.getSerializedSize())
        + dataValue.getSerializedSize();
  }

  @Override
  public int hashCode() {
    return value != null ? value.hashCode() : super.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof IdXmlResourceValue)) {
      return false;
    }
    IdXmlResourceValue other = (IdXmlResourceValue) obj;
    return Objects.equals(value, other.value);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("value", value).toString();
  }

  @Override
  public XmlResourceValue combineWith(XmlResourceValue resourceValue) {
    if (equals(resourceValue)) {
      return this;
    }
    if (resourceValue instanceof IdXmlResourceValue) {
      IdXmlResourceValue otherId = (IdXmlResourceValue) resourceValue;
      if (value == null && otherId.value != null) {
        return otherId;
      }
      if (value != null && otherId.value == null) {
        return this;
      }
    }
    throw new IllegalArgumentException(resourceValue + "is not combinable with " + this);
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
