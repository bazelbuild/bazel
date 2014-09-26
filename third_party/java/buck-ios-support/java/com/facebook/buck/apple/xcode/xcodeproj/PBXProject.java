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
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.dd.plist.NSDictionary;

import java.util.Collections;
import java.util.List;

/**
 * The root object representing the project itself.
 */
public class PBXProject extends PBXContainer {
  private String name;
  private final PBXGroup mainGroup;
  private final List<PBXTarget> targets;
  private final XCConfigurationList buildConfigurationList;
  private final String compatibilityVersion;

  public PBXProject(String name) {
    this.name = Preconditions.checkNotNull(name);
    this.mainGroup = new PBXGroup("mainGroup", null, PBXReference.SourceTree.GROUP);
    this.targets = Lists.newArrayList();
    this.buildConfigurationList = new XCConfigurationList();
    this.compatibilityVersion = "Xcode 3.2";
  }

  public String getName() {
    return name;
  }

  public void setName(String v) {
    name = v;
  }

  public PBXGroup getMainGroup() {
    return mainGroup;
  }

  public List<PBXTarget> getTargets() {
    return targets;
  }

  public XCConfigurationList getBuildConfigurationList() {
    return buildConfigurationList;
  }

  public String getCompatibilityVersion() {
    return compatibilityVersion;
  }

  @Override
  public String isa() {
    return "PBXProject";
  }

  @Override
  public int stableHash() {
    return name.hashCode();
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    s.addField("mainGroup", mainGroup);

    Collections.sort(targets, Ordering.natural().onResultOf(new Function<PBXTarget, String>() {
      @Override
      public String apply(PBXTarget input) {
        return input.getName();
      }
    }));
    s.addField("targets", targets);
    s.addField("buildConfigurationList", buildConfigurationList);
    s.addField("compatibilityVersion", compatibilityVersion);

    NSDictionary d = new NSDictionary();
    d.put("LastUpgradeCheck", "0600");

    s.addField("attributes", d);
  }
}
