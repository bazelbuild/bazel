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

import com.android.aapt.Resources.Item;
import com.android.aapt.Resources.Primitive;
import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.StyledString;
import com.android.aapt.Resources.StyledString.Span;
import com.android.aapt.Resources.Value;
import com.android.resources.ResourceType;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.xml.XmlEscapers;
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
import com.google.devtools.build.android.resources.Visibility;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;
import javax.xml.namespace.QName;

/**
 * Represents a simple Android resource xml value.
 *
 * <p>There is a class of resources that are simple name/value pairs: string
 * (http://developer.android.com/guide/topics/resources/string-resource.html), bool
 * (http://developer.android.com/guide/topics/resources/more-resources.html#Bool), color
 * (http://developer.android.com/guide/topics/resources/more-resources.html#Color), and dimen
 * (http://developer.android.com/guide/topics/resources/more-resources.html#Dimension). These are
 * defined in xml as &lt;<em>resource type</em> name="<em>name</em>" value="<em>value</em>"&gt;. In
 * the interest of keeping the parsing svelte, these are represented by a single class.
 */
@Immutable
public class SimpleXmlResourceValue implements XmlResourceValue {

  static final QName TAG_BOOL = QName.valueOf("bool");
  static final QName TAG_COLOR = QName.valueOf("color");
  static final QName TAG_DIMEN = QName.valueOf("dimen");
  static final QName TAG_DRAWABLE = QName.valueOf("drawable");
  static final QName TAG_FRACTION = QName.valueOf("fraction");
  static final QName TAG_INTEGER = QName.valueOf("integer");
  static final QName TAG_ITEM = QName.valueOf("item");
  static final QName TAG_LAYOUT = QName.valueOf("layout");
  static final QName TAG_MENU = QName.valueOf("menu");
  static final QName TAG_MIPMAP = QName.valueOf("mipmap");
  static final QName TAG_NAVIGATION = QName.valueOf("navigation");
  static final QName TAG_PUBLIC = QName.valueOf("public");
  static final QName TAG_RAW = QName.valueOf("raw");
  static final QName TAG_STRING = QName.valueOf("string");

  /** Provides an enumeration resource type and simple value validation. */
  public enum Type {
    BOOL(TAG_BOOL),
    COLOR(TAG_COLOR),
    DIMEN(TAG_DIMEN),
    DRAWABLE(TAG_DRAWABLE),
    FONT(TAG_ITEM),
    FRACTION(TAG_FRACTION),
    INTEGER(TAG_INTEGER),
    ITEM(TAG_ITEM),
    LAYOUT(TAG_LAYOUT),
    MENU(TAG_MENU),
    MIPMAP(TAG_MIPMAP),
    NAVIGATION(TAG_NAVIGATION),
    PUBLIC(TAG_PUBLIC),
    RAW(TAG_RAW),
    STRING(TAG_STRING);
    private final QName tagName;

    Type(QName tagName) {
      this.tagName = tagName;
    }

    public static Type from(ResourceType resourceType) {
      for (Type valueType : values()) {
        if (valueType.tagName.getLocalPart().equals(resourceType.getName())) {
          return valueType;
        } else if (resourceType.getName().equalsIgnoreCase(valueType.name())) {
          return valueType;
        }
      }
      throw new IllegalArgumentException(
          String.format(
              "%s resource type not found in available types: %s",
              resourceType, Arrays.toString(values())));
    }
  }

  private final Visibility visibility;
  private final Item item;
  private final ImmutableMap<String, String> attributes;
  // TODO(b/112848607): remove untyped "value" String in favor of "item" above.
  @Nullable private final String value;
  private final Type valueType;

  public static XmlResourceValue createWithValue(Type valueType, String value) {
    return of(valueType, ImmutableMap.<String, String>of(), value);
  }

  public static XmlResourceValue withAttributes(
      Type valueType, ImmutableMap<String, String> attributes) {
    return of(valueType, attributes, null);
  }

  public static XmlResourceValue itemWithFormattedValue(
      ResourceType resourceType, String format, String value) {
    return of(Type.ITEM, ImmutableMap.of("type", resourceType.getName(), "format", format), value);
  }

  public static XmlResourceValue itemWithValue(ResourceType resourceType, String value) {
    return of(Type.ITEM, ImmutableMap.of("type", resourceType.getName()), value);
  }

  public static XmlResourceValue itemPlaceHolderFor(ResourceType resourceType) {
    return withAttributes(Type.ITEM, ImmutableMap.of("type", resourceType.getName()));
  }

  public static XmlResourceValue of(
      Type valueType, ImmutableMap<String, String> attributes, @Nullable String value) {
    return new SimpleXmlResourceValue(
        Visibility.UNKNOWN, valueType, Item.getDefaultInstance(), attributes, value);
  }

  private SimpleXmlResourceValue(
      Visibility visibility,
      Type valueType,
      Item item,
      ImmutableMap<String, String> attributes,
      String value) {
    this.visibility = visibility;
    this.valueType = valueType;
    this.item = item;
    this.value = value;
    this.attributes = attributes;
  }

  @Override
  public void write(
      FullyQualifiedName key, DataSource source, AndroidDataWritingVisitor mergedDataWriter) {

    StartTag startTag =
        mergedDataWriter
            .define(key)
            .derivedFrom(source)
            .startTag(valueType.tagName)
            .named(key)
            .addAttributesFrom(attributes.entrySet());

    if (value != null) {
      startTag.closeTag().addCharactersOf(value).endTag().save();
    } else {
      startTag.closeUnaryTag().save();
    }
  }

  @SuppressWarnings("deprecation")
  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(
        Type.valueOf(proto.getValueType()),
        ImmutableMap.copyOf(proto.getAttributeMap()),
        proto.hasValue() ? proto.getValue() : null);
  }

  public static XmlResourceValue from(
      Value proto, Visibility visibility, ResourceType resourceType) {
    Item item = proto.getItem();
    String stringValue = null;
    ImmutableMap.Builder<String, String> attributes = ImmutableMap.builder();

    if (item.hasStr()) {
      stringValue = XmlEscapers.xmlContentEscaper().escape(item.getStr().getValue());
    } else if (item.hasRef()) {
      stringValue = "@" + item.getRef().getName();
      attributes.put("format", "reference");
    } else if (item.hasStyledStr()) {
      StyledString styledString = item.getStyledStr();
      StringBuilder stringBuilder = new StringBuilder(styledString.getValue());

      for (Span span : styledString.getSpanList()) {
        stringBuilder.append(
            String.format(";%s,%d,%d", span.getTag(), span.getFirstChar(), span.getLastChar()));
      }
      stringValue = stringBuilder.toString();
    } else if (item.hasPrim()) {
      stringValue = convertPrimitiveToString(item.getPrim());
    } else {
      throw new IllegalArgumentException(
          String.format("'%s' with value %s is not a simple resource type.", resourceType, proto));
    }

    return new SimpleXmlResourceValue(
        visibility,
        Type.valueOf(resourceType.toString().toUpperCase(Locale.ENGLISH)),
        item,
        attributes.build(),
        stringValue);
  }

  static String convertPrimitiveToString(Primitive primitive) {
    switch (primitive.getOneofValueCase()) {
      case NULL_VALUE:
        return "(null)";
      case EMPTY_VALUE:
        return "(empty)";
      case FLOAT_VALUE:
        return Float.toString(primitive.getFloatValue());
      case INT_DECIMAL_VALUE:
        return Integer.toString(primitive.getIntDecimalValue());
      case INT_HEXADECIMAL_VALUE:
        return String.format(Locale.ROOT, "0x%x", primitive.getIntHexadecimalValue());
      case BOOLEAN_VALUE:
        return Boolean.toString(primitive.getBooleanValue());
      case DIMENSION_VALUE:
        return ComplexConverter.convertComplexPrimitiveToString(
            primitive.getDimensionValue(), /*isDimension=*/ true);
      case FRACTION_VALUE:
        return ComplexConverter.convertComplexPrimitiveToString(
            primitive.getFractionValue(), /*isDimension=*/ false);

        // Rendering all colors as normalized 32-bit hex.  No one cares how they were written in the
        // original XML, and rendering these differently would lead to pointless resouce conflicts
        // reported if e.g. one XML file uses #123 while another uses #112233.
      case COLOR_ARGB8_VALUE:
        return String.format(Locale.ROOT, "#%08X", primitive.getColorArgb8Value());
      case COLOR_RGB8_VALUE:
        return String.format(Locale.ROOT, "#%08X", primitive.getColorRgb8Value());
      case COLOR_ARGB4_VALUE:
        return String.format(Locale.ROOT, "#%08X", primitive.getColorArgb4Value());
      case COLOR_RGB4_VALUE:
        return String.format(Locale.ROOT, "#%08X", primitive.getColorRgb4Value());

      case DIMENSION_VALUE_DEPRECATED:
      case FRACTION_VALUE_DEPRECATED:
        // we don't expect to deserialize data from older aapt2 builds
      case ONEOFVALUE_NOT_SET:
        break;
    }
    throw new IllegalArgumentException("Invalid primitive value " + primitive);
  }

  // See 'print_complex' defined in:
  // https://android.googlesource.com/platform/frameworks/base/+/master/libs/androidfw/ResourceTypes.cpp
  private static final class ComplexConverter {
    static final String[] DIMENSION_TYPE_STRINGS =
        new String[] {"px", "dp", "sp", "pt", "in", "mm"};
    static final String[] FRACTION_TYPE_STRINGS = new String[] {"%", "%p"};
    static final int[] RADIX_SHIFTS = new int[] {0, 7, 15, 23};

    static String convertComplexPrimitiveToString(int rawValue, boolean isDimension) {
      String typeString;
      try {
        if (isDimension) {
          typeString = DIMENSION_TYPE_STRINGS[rawValue & 0xF];
        } else {
          typeString = FRACTION_TYPE_STRINGS[rawValue & 0xF];
        }
      } catch (IndexOutOfBoundsException e) {
        typeString = " (unknown unit)";
      }

      int radixIdx = (rawValue >> 4) & 0x3;
      float value = (rawValue >> 8) * (1.0f / (1 << RADIX_SHIFTS[radixIdx]));

      // Use Float.toString instead of String.format("%f") to avoid excessive trailing zeros.
      // Strings should never be localized anyway.
      // * https://stackoverflow.com/a/44202755
      // * https://issuetracker.google.com/issues/64962882
      return String.format(Locale.ROOT, "%s%s", Float.toString(value), typeString);
    }
  }

  @Override
  public void writeResourceToClass(
      DependencyInfo dependencyInfo, FullyQualifiedName key, AndroidResourceSymbolSink sink) {
    sink.acceptSimpleResource(dependencyInfo, visibility, key.type(), key.name());
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
            .setType(SerializeFormat.DataValueXml.XmlType.SIMPLE)
            // TODO(corysmith): Find a way to avoid writing strings to the serialized format
            // it's inefficient use of space and costs more when deserializing.
            .putAllAttribute(attributes);
    if (value != null) {
      xmlValueBuilder.setValue(value);
    }
    builder.setXmlValue(xmlValueBuilder.setValueType(valueType.name()));
    return XmlResourceValues.serializeProtoDataValue(output, builder);
  }

  @Override
  public int hashCode() {
    return Objects.hash(visibility, valueType, attributes, value);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof SimpleXmlResourceValue)) {
      return false;
    }
    SimpleXmlResourceValue other = (SimpleXmlResourceValue) obj;
    return Objects.equals(visibility, other.visibility)
        && Objects.equals(valueType, other.valueType)
        && Objects.equals(attributes, other.attributes)
        // TODO(b/112848607): include the "item" proto in comparison; right now it's redundant.
        && Objects.equals(value, other.value);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("valueType", valueType)
        .add("attributes", attributes)
        .add("value", value)
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
    if (value != null) {
      return String.format(" %s (with value %s)", source.asConflictString(), value);
    }
    return source.asConflictString();
  }

  @Override
  public Visibility getVisibility() {
    return visibility;
  }

  @Override
  public ImmutableList<Reference> getReferencedResources() {
    return item.hasRef() ? ImmutableList.of(item.getRef()) : ImmutableList.of();
  }
}
