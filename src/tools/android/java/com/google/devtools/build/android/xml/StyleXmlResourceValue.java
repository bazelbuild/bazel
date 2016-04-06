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

import java.nio.file.Path;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;

import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;

/**
 * Represents an Android Style Resource.
 *
 * <p>Styles (http://developer.android.com/guide/topics/resources/style-resource.html) define a
 * look and feel for a layout or other ui construct. They are effectively a s set of values that
 * correspond to &lt;attr&gt; resources defined either in the base android framework or in other
 * resources. They also allow inheritance on other styles. For a style to valid in a given resource
 * pass, they must only contain definer attributes with acceptable values.
 * <code>
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
  public static final Function<Entry<String, String>, String> ENTRY_TO_ITEM =
      new Function<Entry<String, String>, String>() {
        @Nullable
        @Override
        public String apply(Entry<String, String> input) {
          return String.format("<item name='%s'>%s</item>", input.getKey(), input.getValue());
        }
      };
  private final String parent;
  private final ImmutableMap<String, String> values;

  public static StyleXmlResourceValue of(String parent, Map<String, String> values) {
    return new StyleXmlResourceValue(parent, ImmutableMap.copyOf(values));
  }

  private StyleXmlResourceValue(String parent, ImmutableMap<String, String> values) {
    this.parent = parent;
    this.values = values;
  }

  @Override
  public void write(
      FullyQualifiedName key, Path source, AndroidDataWritingVisitor mergedDataWriter) {
    mergedDataWriter.writeToValuesXml(
        key,
        FluentIterable.from(
                ImmutableList.of(
                    String.format("<!-- %s -->", source),
                    parent == null || parent.isEmpty()
                        ? String.format("<style name='%s'>", key.name())
                        : String.format("<style name='%s' parent='%s'>", key.name(), parent)))
            .append(FluentIterable.from(values.entrySet()).transform(ENTRY_TO_ITEM))
            .append("</style>"));
  }

  @Override
  public int hashCode() {
    return Objects.hash(parent, values);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof StyleXmlResourceValue)) {
      return false;
    }
    StyleXmlResourceValue other = (StyleXmlResourceValue) obj;
    return Objects.equals(parent, other.parent) && Objects.equals(values, other.values);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass())
        .add("parent", parent)
        .add("values", values)
        .toString();
  }
}
