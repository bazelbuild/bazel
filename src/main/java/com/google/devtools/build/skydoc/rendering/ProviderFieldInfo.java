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

/**
 * Stores information about a provider field definition.
 */
public class ProviderFieldInfo {
  private final String name;
  private final String docString;

  public ProviderFieldInfo(String name) {
    this(name, "(Undocumented)");
  }

  public ProviderFieldInfo(String name, String docString) {
    this.name = name;
    this.docString = docString.trim();
  }

  @SuppressWarnings("unused") // Used by markdown template.
  public String getName() {
    return name;
  }

  @SuppressWarnings("unused") // Used by markdown template.
  public String getDocString() {
    return docString;
  }
}
