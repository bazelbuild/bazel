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
import com.google.common.base.Optional;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Lists;

import java.util.Comparator;
import java.util.Collections;
import java.util.List;

/**
 * List of build configurations.
 */
public class XCConfigurationList extends PBXProjectItem {
  private List<XCBuildConfiguration> buildConfigurations;
  private Optional<String> defaultConfigurationName;
  private boolean defaultConfigurationIsVisible;

  private final LoadingCache<String, XCBuildConfiguration> buildConfigurationsByName;

  public XCConfigurationList() {
    buildConfigurations = Lists.newArrayList();
    defaultConfigurationName = Optional.absent();
    defaultConfigurationIsVisible = false;

    buildConfigurationsByName = CacheBuilder.newBuilder().build(
        new CacheLoader<String, XCBuildConfiguration>() {
          @Override
          public XCBuildConfiguration load(String key) throws Exception {
            XCBuildConfiguration configuration = new XCBuildConfiguration(key);
            buildConfigurations.add(configuration);
            return configuration;
          }
        });
  }

  public LoadingCache<String, XCBuildConfiguration> getBuildConfigurationsByName() {
    return buildConfigurationsByName;
  }

  @Override
  public String isa() {
    return "XCConfigurationList";
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    Collections.sort(buildConfigurations, new Comparator<XCBuildConfiguration>() {
      @Override
      public int compare(XCBuildConfiguration o1, XCBuildConfiguration o2) {
        return o1.getName().compareTo(o2.getName());
      }
    });
    s.addField("buildConfigurations", buildConfigurations);

    if (defaultConfigurationName.isPresent()) {
      s.addField("defaultConfigurationName", defaultConfigurationName.get());
    }
    s.addField("defaultConfigurationIsVisible", defaultConfigurationIsVisible);
  }
}
