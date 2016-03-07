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
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.XmlResourceValue;

import java.io.Writer;

/**
 * Represents an Android Resource id.
 *
 * <p>Ids (http://developer.android.com/guide/topics/resources/more-resources.html#Id) are
 * special -- unlike other resources they cannot be overridden. This is due to the
 * nature of their usage. Each id corresponds to context sensitive resource of component, meaning
 * that they have no intrinsic defined value. They exist to reference parts of other resources.
 * Ids can also be declared on the fly in components with the syntax @[+][package:]id/resource_name.
 */
public class IdXmlResourceValue implements XmlResourceValue {

  static final IdXmlResourceValue SINGLETON = new IdXmlResourceValue();

  public static XmlResourceValue of() {
    return SINGLETON;
  }

  @Override
  public void write(Writer buffer, FullyQualifiedName name) {
    // TODO(corysmith): Implement write.
    throw new UnsupportedOperationException();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).toString();
  }
}
