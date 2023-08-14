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

import com.google.devtools.build.lib.starlarkbuildapi.StarlarkAttrModuleApi.Descriptor;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderNameGroup;
import java.util.List;
import java.util.Optional;
import net.starlark.java.eval.Printer;

/**
 * Fake implementation of {@link Descriptor}.
 */
public class FakeDescriptor implements Descriptor {
  private final AttributeType type;
  private final Optional<String> docString;
  private final boolean mandatory;
  private final List<List<String>> providerNameGroups;
  private final String defaultRepresentation;

  public FakeDescriptor(
      AttributeType type,
      Optional<String> docString,
      boolean mandatory,
      List<List<String>> providerNameGroups,
      Object defaultObject) {
    this.type = type;
    this.docString = docString;
    this.mandatory = mandatory;
    this.providerNameGroups = providerNameGroups;
    this.defaultRepresentation = defaultObject.toString();
  }

  @Override
  public void repr(Printer printer) {}

  public AttributeInfo asAttributeInfo(String attributeName) {
    AttributeInfo.Builder attrInfo =
        AttributeInfo.newBuilder()
            .setName(attributeName)
            .setType(type)
            .setMandatory(mandatory)
            .setDefaultValue(mandatory ? "" : defaultRepresentation);
    docString.ifPresent(attrInfo::setDocString);

    if (!providerNameGroups.isEmpty()) {
      for (List<String> providerNameGroup : providerNameGroups) {
        ProviderNameGroup.Builder providerNameListBuild = ProviderNameGroup.newBuilder();
        ProviderNameGroup providerNameList =
            providerNameListBuild.addAllProviderName(providerNameGroup).build();
        attrInfo.addProviderNameGroup(providerNameList);
      }
    }
    return attrInfo.build();
  }
}
