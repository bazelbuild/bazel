// Copyright 2009 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import com.google.auto.value.AutoValue;
import java.util.EnumMap;
import java.util.Map;

/** TestIntegration represents an external link that is integrated with the test results. */
@AutoValue
public abstract class TestIntegration {

  /** Represents each available field for TestIntegration. */
  public enum ExternalLinkAttribute {
    NAME,
    URL,
    CONTACT_EMAIL,
    COMPONENT_ID,
    DESCRIPTION,
    ICON_NAME,
    ICON_URL,
    BACKGROUND_COLOR,
    FOREGROUND_COLOR;

    /** Gets the string representation of the current enum. */
    public String getXmlAttributeName() {
      return name().toLowerCase();
    }
  }

  // Group or user name responsible for this external integration.
  abstract String contactEmail();
  // Component id (numeric) for this external integration.
  abstract String componentId();
  // Display name of this external integration.
  abstract String name();
  // URL that will display more data about this test result or integration.
  abstract String url();
  // Optional: URL or name of the icon to be displayed.
  abstract String iconUrl();

  abstract String iconName();
  // Optional: Textual description that shows up as tooltip.
  abstract String description();
  // Optional: Foreground color.
  abstract String foregroundColor();
  // Optional: Background color.
  abstract String backgroundColor();

  static Builder builder() {
    return new AutoValue_TestIntegration.Builder()
        .setIconName("")
        .setIconUrl("")
        .setDescription("")
        .setForegroundColor("")
        .setBackgroundColor("");
  }

  @AutoValue.Builder
  abstract static class Builder {
    public abstract Builder setContactEmail(String email);

    public abstract Builder setComponentId(String id);

    public abstract Builder setName(String name);

    public abstract Builder setUrl(String url);

    public abstract Builder setIconUrl(String iconUrl);

    public abstract Builder setIconName(String iconName);

    public abstract Builder setDescription(String description);

    public abstract Builder setForegroundColor(String foregroundColor);

    public abstract Builder setBackgroundColor(String backgroundColor);

    abstract TestIntegration build();
  }

  /*
   * getAttributeValueMap returns all of this TestIntegration's values in a Map.
   */
  public Map<ExternalLinkAttribute, String> getAttributeValueMap() {
    Map<ExternalLinkAttribute, String> map = new EnumMap<>(ExternalLinkAttribute.class);
    map.put(ExternalLinkAttribute.NAME, name());
    map.put(ExternalLinkAttribute.URL, url());
    map.put(ExternalLinkAttribute.CONTACT_EMAIL, contactEmail());
    map.put(ExternalLinkAttribute.COMPONENT_ID, componentId());
    map.put(ExternalLinkAttribute.DESCRIPTION, description());
    map.put(ExternalLinkAttribute.ICON_NAME, iconName());
    map.put(ExternalLinkAttribute.ICON_URL, iconUrl());
    map.put(ExternalLinkAttribute.BACKGROUND_COLOR, backgroundColor());
    map.put(ExternalLinkAttribute.FOREGROUND_COLOR, foregroundColor());
    return map;
  }
}
