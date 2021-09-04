// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.android.aapt.Resources.MacroBody;
import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.Value;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidDataWritingVisitor.StartTag;
import com.google.devtools.build.android.AndroidResourceSymbolSink;
import com.google.devtools.build.android.DataSource;
import com.google.devtools.build.android.DependencyInfo;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;
import com.google.devtools.build.android.resources.Visibility;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Objects;

/**
 * Represents an Android Macro resource.
 *
 * <p>Macros are compile-time resource definitions that have their contents substituted wherever
 * they are referenced in xml. For example: <code>
 *   &lt;macro name="is_enabled"&gt;true&lt;/macro&gt;
 *   &lt;bool name="is_prod"&gt;&commat;macro/is_enabled&lt;/macro&gt;
 * </code> The contents of the macro above will be substituted in place of the macro reference,
 * resulting a resource table containing: <code>
 *   &lt;bool name="is_prod"&gt;true&lt;/macro&gt;
 * </code>
 */
public class MacroXmlResourceValue implements XmlResourceValue {
  private final String rawString;

  private MacroXmlResourceValue(String rawString) {
    this.rawString = rawString;
  }

  public static XmlResourceValue of(String rawContents) {
    return new MacroXmlResourceValue(rawContents);
  }

  public static XmlResourceValue from(Value proto, Visibility visibility) {
    MacroBody macro = proto.getCompoundValue().getMacro();
    return new MacroXmlResourceValue(macro.getRawString());
  }

  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return new MacroXmlResourceValue(proto.getValue());
  }

  /**
   * Each XmlValue is expected to write a valid representation in xml to the writer.
   *
   * @param key The FullyQualified name for the xml resource being written.
   * @param source The source of the value to allow for proper comment annotation.
   * @param mergedDataWriter The target writer.
   */
  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {
    StartTag startTag =
        mergedDataWriter.define(key).derivedFrom(source).startTag("macro").named(key);
    if (rawString == null) {
      startTag.closeUnaryTag().save();
    } else {
      startTag.closeTag().addCharactersOf(rawString).endTag().save();
    }
  }

  /** Serializes the resource value to the OutputStream and returns the bytes written. */
  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream out) throws IOException {
    DataValueXml.Builder xmlValue =
        SerializeFormat.DataValueXml.newBuilder()
            .setType(XmlType.MACRO)
            .putAllNamespace(namespaces.asMap());
    if (rawString != null) {
      xmlValue.setValue(rawString);
    }
    return XmlResourceValues.serializeProtoDataValue(
        out, XmlResourceValues.newSerializableDataValueBuilder(sourceId).setXmlValue(xmlValue));
  }

  /**
   * Combines these xml values together and returns a single value.
   *
   * @throws IllegalArgumentException always since macros are not a combinable resource
   */
  @Override
  public XmlResourceValue combineWith(XmlResourceValue value) {
    throw new IllegalArgumentException(this + " is not a combinable resource.");
  }

  /**
   * Macros cannot be merged with any xml values, so this method always returns 0 which indicates
   * this and the other value have equal priority.
   */
  @Override
  public int compareMergePriorityTo(XmlResourceValue value) {
    return 0;
  }

  /**
   * Queue up writing the resource to the given {@link AndroidResourceClassWriter}. Each resource
   * can generate one or more (in the case of styleable) fields and inner classes in the R class.
   *
   * @param dependencyInfo The provenance (in terms of Bazel relationship) of the resource
   * @param key The FullyQualifiedName of the resource
   * @param sink the symbol sink for producing source and classes
   */
  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {}

  /** Returns a representation of the xml value as a string suitable for conflict messages. */
  @Override
  public String asConflictStringWith(DataSource source) {
    return source.asConflictString();
  }

  /** Visibility of this resource as denoted by a {@code <public>} tag, or lack thereof. */
  @Override
  public Visibility getVisibility() {
    return Visibility.UNKNOWN;
  }

  /** Resources referenced via XML attributes or proxying resource definitions. */
  @Override
  public ImmutableList<Reference> getReferencedResources() {
    return ImmutableList.of();
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(rawString);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof MacroXmlResourceValue)) {
      return false;
    }
    MacroXmlResourceValue other = (MacroXmlResourceValue) obj;
    return Objects.equals(rawString, other.rawString);
  }
}
