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
import com.google.common.base.MoreObjects;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;

import java.io.Writer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Represent an Android styleable resource.
 *
 * <p>Styleable resources are groups of attributes that can be applied to views. They are, for the
 * most part, vaguely documented (http://developer.android.com/training/custom-views/create-view
 * .html#customattr). It's worth noting that attributes declared inside a
 * &lt;declare-styleable&gt; tags, for example;
 * <code>
 *  <declare-styleable name="PieChart">
 *     <attr name="showText" format="boolean" />
 *  </declare-styleable>
 * </code>
 *
 * Can also be seen as:
 * <code>
 *  <attr name="showText" format="boolean" />
 *  <declare-styleable name="PieChart">
 *     <attr name="showText"/>
 *  </declare-styleable>
 * </code>
 *
 * <p>The StyleableXmlValue only contains names of the attributes it holds, not definitions.
 */
public class StyleableXmlResourceValue implements XmlResourceValue {
  private final List<String> attrs;

  private StyleableXmlResourceValue(List<String> attrs) {
    this.attrs = attrs;
  }

  public static XmlResourceValue of(List<String> attrs) {
    return new StyleableXmlResourceValue(attrs);
  }

  @VisibleForTesting
  public static XmlResourceValue of(String... attrs) {
    return new StyleableXmlResourceValue(Arrays.asList(attrs));
  }

  @Override
  public void write(Writer buffer, FullyQualifiedName name) {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
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
