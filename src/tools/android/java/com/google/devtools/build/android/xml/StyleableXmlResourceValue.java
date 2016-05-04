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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.MoreObjects;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import com.google.devtools.build.android.proto.SerializeFormat;
import com.google.devtools.build.android.proto.SerializeFormat.DataValueXml.XmlType;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Represent an Android styleable resource.
 *
 * <p>
 * Styleable resources are groups of attributes that can be applied to views. They are, for the most
 * part, vaguely documented (http://developer.android.com/training/custom-views/create-view
 * .html#customattr). It's worth noting that attributes declared inside a &lt;declare-styleable&gt;
 * tags, for example; <code>
 *  <declare-styleable name="PieChart">
 *     <attr name="showText" format="boolean" />
 *  </declare-styleable>
 * </code>
 *
 * Can also be seen as: <code>
 *  <attr name="showText" format="boolean" />
 *  <declare-styleable name="PieChart">
 *     <attr name="showText"/>
 *  </declare-styleable>
 * </code>
 *
 * <p>
 * The StyleableXmlValue only contains names of the attributes it holds, not definitions.
 */
@Immutable
public class StyleableXmlResourceValue implements XmlResourceValue {
  public static final Function<String, String> ITEM_TO_ATTR =
      new Function<String, String>() {
        @Nullable
        @Override
        public String apply(@Nullable String input) {
          return String.format("<attr name='%s'/>", input);
        }
      };
  private final ImmutableList<String> attrs;

  private StyleableXmlResourceValue(ImmutableList<String> attrs) {
    this.attrs = attrs;
  }

  public static XmlResourceValue of(List<String> attrs) {
    return new StyleableXmlResourceValue(ImmutableList.copyOf(attrs));
  }

  @VisibleForTesting
  public static XmlResourceValue of(String... attrs) {
    return new StyleableXmlResourceValue(
        Ordering.natural().immutableSortedCopy(Arrays.asList(attrs)));
  }

  @Override
  public void write(
      FullyQualifiedName key, Path source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.writeToValuesXml(
        key,
        FluentIterable.from(
                ImmutableList.of(
                    String.format("<!-- %s -->", source),
                    String.format("<declare-styleable name='%s'>", key.name())))
            .append(FluentIterable.from(attrs).transform(ITEM_TO_ATTR))
            .append("</declare-styleable>"));
  }

  @Override
  public int serializeTo(Path source, OutputStream output) throws IOException {
    return XmlResourceValues.serializeProtoDataValue(
        output,
        XmlResourceValues.newSerializableDataValueBuilder(source)
            .setXmlValue(
                SerializeFormat.DataValueXml.newBuilder()
                    .setType(XmlType.STYLEABLE)
                    .addAllListValue(attrs)));
  }

  public static XmlResourceValue from(SerializeFormat.DataValueXml proto) {
    return of(proto.getListValueList());
  }

  @Override
  public int hashCode() {
    return attrs.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof StyleableXmlResourceValue)) {
      return false;
    }
    StyleableXmlResourceValue other = (StyleableXmlResourceValue) obj;
    return Objects.equals(attrs, other.attrs);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("attrs", attrs).toString();
  }
}
