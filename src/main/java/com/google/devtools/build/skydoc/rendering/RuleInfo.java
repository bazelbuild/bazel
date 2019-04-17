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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import java.util.Collection;

/**
 * Stores information about a skylark rule definition.
 */
public class RuleInfo {

  private final BaseFunction identifierFunction;
  private final Location location;
  private final String docString;
  private final Collection<AttributeInfo> attrInfos;

  public RuleInfo(BaseFunction identifierFunction,
      Location location,
      String docString,
      Collection<AttributeInfo> attrInfos) {
    this.identifierFunction = identifierFunction;
    this.location = location;
    this.docString = docString;
    this.attrInfos = attrInfos;
  }

  public BaseFunction getIdentifierFunction() {
    return identifierFunction;
  }

  public Location getLocation() {
    return location;
  }

  public String getDocString() {
    return docString;
  }

  public Collection<AttributeInfo> getAttributes() {
    return attrInfos;
  }
}
