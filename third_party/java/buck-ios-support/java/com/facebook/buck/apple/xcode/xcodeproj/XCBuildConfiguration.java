/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode.xcodeproj;

import com.facebook.buck.apple.xcode.XcodeprojSerializer;

/**
 * Build configuration containing a file reference ton an xcconfig file and additional inline
 * settings.
 */
public class XCBuildConfiguration extends PBXBuildStyle {
  private PBXFileReference baseConfigurationReference;

  public XCBuildConfiguration(String name) {
    super(name);
  }

  public PBXFileReference getBaseConfigurationReference() {
    return baseConfigurationReference;
  }
  public void setBaseConfigurationReference(PBXFileReference v) {
    baseConfigurationReference = v;
  }

  @Override
  public String isa() {
    return "XCBuildConfiguration";
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    s.addField("baseConfigurationReference", baseConfigurationReference);
  }
}
