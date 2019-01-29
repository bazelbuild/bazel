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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAttrApi.Descriptor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.skydoc.rendering.AttributeInfo.Type;

/**
 * Fake implementation of {@link Descriptor}.
 */
public class FakeDescriptor implements Descriptor {
  private final Type type;
  private final String docString;
  private final boolean mandatory;

  public FakeDescriptor(Type type, String docString, boolean mandatory) {
    this.type = type;
    this.docString = docString;
    this.mandatory = mandatory;
  }

  public Type getType() {
    return type;
  }

  public String getDocString() {
    return docString;
  }

  public boolean isMandatory() {
    return mandatory;
  }

  @Override
  public void repr(SkylarkPrinter printer) {}
}
