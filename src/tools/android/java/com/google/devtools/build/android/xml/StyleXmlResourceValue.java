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

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.Style;
import com.android.aapt.Resources.Value;
import com.google.common.base.Function;
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
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;
import com.google.devtools.build.android.resources.Visibility;
import java.io.IOException;
import java.io.OutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Represents an Android Style Resource.
 *
 * <p>
 * Styles (http://developer.android.com/guide/topics/resources/style-resource.html) define a look
 * and feel for a layout or other ui construct. They are effectively a s set of values that
 * correspond to &lt;attr&gt; resources defined either in the base android framework or in other
 * resources. They also allow inheritance on other styles. For a style to valid in a given resource
 * pass, they must only contain definer attributes with acceptable values. <code>
 *   &lt;resources&gt;
 *     &lt;style name="CustomText" parent="@style/Text"&gt;
 *       &lt;item name="android:textSize"&gt;20sp&lt;/item&gt;
 *       &lt;item name="android:textColor"&gt;#008&lt;/item&gt;
 *     &lt;/style&gt;
 *  &lt;/resources&gt;
 * </code>
 */
@Immutable
public class StyleXmlResourceValue implements XmlResourceValue {
  public static final Function<Map.Entry<String, String>, String> ENTRY_TO_ITEM =
      new Function<Map.Entry<String, String>, String>() {
        @Nullable
        @Override
        public String apply(Map.Entry<String, String> input) {
          return String.format("<item name='%s'>%s</item>", input.getKey(), input.getValue());
        }
      };

  private final Visibility visibility;
  private final Style style;
  // TODO(b/112848607): remove parent/values in favor of "style" above, or replace the Strings with
  // stronger types.
  private final String parent;
  private final ImmutableMap<String, String> values;

  public static StyleXmlResourceValue of(String parent, Map<String, String> values) {
    return new StyleXmlResourceValue(
        Visibility.UNKNOWN, Style.getDefaultInstance(), parent, ImmutableMap.copyOf(values));
  }

  @SuppressWarnings("deprecation")
  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(proto.hasValue() ? proto.getValue() : null, proto.getMappedStringValue());
  }

  public static XmlResourceValue from(Value proto, Visibility visibility) {
    Style style = proto.getCompoundValue().getStyle();
    String parent = "";

    if (style.hasParent()) {
      parent = proto.getCompoundValue().getStyle().getParent().getName();
      if (parent.startsWith("style/")) {
        // Aapt2 compile breaks when style parent references are prepended with 'style/'
        parent = parent.substring(6);
      }
    }

    Map<String, String> items = itemMapFromProto(style);

    return new StyleXmlResourceValue(visibility, style, parent, ImmutableMap.copyOf(items));
  }

  private StyleXmlResourceValue(
      Visibility visibility,
      Style style,
      @Nullable String parent,
      ImmutableMap<String, String> values) {
    this.visibility = visibility;
    this.style = style;
    this.parent = parent;
    this.values = values;
  }

  private static Map<String, String> itemMapFromProto(Style style) {
    Map<String, String> result = new LinkedHashMap<>();

    for (Style.Entry styleEntry : style.getEntryList()) {
      String itemName = styleEntry.getKey().getName().replace("attr/", "");
      String itemValue;

      if (styleEntry.getItem().hasRawStr()) {
        itemValue = styleEntry.getItem().getRawStr().getValue();
      } else if (styleEntry.getItem().hasRef()) {
        itemValue = "@" + styleEntry.getItem().getRef().getName();
        if (itemValue.equals("@")) {
          itemValue = "@null";
        }
      } else {
        itemValue = styleEntry.getItem().getStr().getValue();
      }
      result.put(itemName, itemValue);
    }

    return result;
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {

    ValuesResourceDefinition definition =
        mergedDataWriter
            .define(key)
            .derivedFrom(source)
            .startTag("style")
            .named(key)
            .optional()
            .attribute("parent")
            .setTo(parent)
            .closeTag()
            .addCharactersOf("\n");
    for (Map.Entry<String, String> entry : values.entrySet()) {
      definition =
          definition
              .startItemTag()
              .named(entry.getKey())
              .closeTag()
              .addCharactersOf(entry.getValue())
              .endTag()
              .addCharactersOf("\n");
    }
    definition.endTag().save();
  }

  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(dependencyInfo, visibility, key.type(), key.name());
  }

  @Override
  public int serializeTo(int sourceId, Namespaces namespaces, OutputStream output)
      throws IOException {
    SerializeFormat.DataValueXml.Builder xmlValueBuilder =
        SerializeFormat.DataValueXml.newBuilder()
            .setType(XmlType.STYLE)
            .putAllNamespace(namespaces.asMap())
            .putAllMappedStringValue(values);
    if (parent != null) {
      xmlValueBuilder.setValue(parent);
    }
    return XmlResourceValues.serializeProtoDataValue(
        output,
        XmlResourceValues.newSerializableDataValueBuilder(sourceId).setXmlValue(xmlValueBuilder));
  }

  @Override
  public int hashCode() {
    return Objects.hash(visibility, parent, values);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof StyleXmlResourceValue)) {
      return false;
    }
    StyleXmlResourceValue other = (StyleXmlResourceValue) obj;
    return Objects.equals(visibility, other.visibility)
        && Objects.equals(parent, other.parent)
        // TODO(b/112848607): include the "style" proto in comparison; right now it's redundant.
        && Objects.equals(values, other.values);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("parent", parent)
        .add("values", values)
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
  public String asConflictStringWith(DataSource source) {
    return source.asConflictString();
  }

  @Override
  public Visibility getVisibility() {
    return visibility;
  }

  @Override
  public ImmutableList<Reference> getReferencedResources() {
    ImmutableList.Builder<Reference> result = ImmutableList.builder();
    if (style.hasParent()) {
      result.add(style.getParent());
    }
    for (Style.Entry entry : style.getEntryList()) {
      result.add(entry.getKey());
      if (entry.getItem().hasRef()) {
        result.add(entry.getItem().getRef());
      }
    }
    return result.build();
  }
}
