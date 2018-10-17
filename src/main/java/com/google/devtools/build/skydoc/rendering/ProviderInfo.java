// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import com.google.devtools.build.lib.syntax.BaseFunction;
import java.util.Collection;

/**
 * Stores information about a starlark provider definition.
 *
 * For example, in <pre>FooInfo = provider(doc = 'My provider', fields = {'bar' : 'a bar'})</pre>,
 * this contains all information about the definition of FooInfo for purposes of generating its
 * documentation.
 */
public class ProviderInfo {

  private final BaseFunction identifier;
  private final String docString;
  private final Collection<ProviderFieldInfo> fieldInfos;

  public ProviderInfo(BaseFunction identifier,
      String docString,
      Collection<ProviderFieldInfo> fieldInfos) {
    this.identifier = identifier;
    this.docString = docString;
    this.fieldInfos = fieldInfos;
  }

  public BaseFunction getIdentifier() {
    return identifier;
  }

  public String getDocString() {
    return docString;
  }

  public Collection<ProviderFieldInfo> getFields() {
    return fieldInfos;
  }
}
