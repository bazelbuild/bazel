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

import java.util.EnumMap;
import java.util.Map;

/** TestIntegration represents an external link that is integrated with the test results. */
public class TestIntegration {

  private final String contactEmail;
  private final String componentId;
  private final String name;
  private final String url;
  private final String iconUrl;
  private final String iconName;
  private final String description;
  private final String foregroundColor;
  private final String backgroundColor;

  public TestIntegration(
      String contactEmail,
      String componentId,
      String name,
      String url,
      String iconUrl,
      String iconName,
      String description,
      String foregroundColor,
      String backgroundColor) {
    this.contactEmail = contactEmail;
    this.componentId = componentId;
    this.name = name;
    this.url = url;
    this.iconUrl = iconUrl;
    this.iconName = iconName;
    this.description = description;
    this.foregroundColor = foregroundColor;
    this.backgroundColor = backgroundColor;
  }

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
  String contactEmail() {
    return contactEmail;
  }

  // Component id (numeric) for this external integration.
  String componentId() {
    return componentId;
  }

  // Display name of this external integration.
  String name() {
    return name;
  }

  // URL that will display more data about this test result or integration.
  String url() {
    return url;
  }

  // Optional: URL or name of the icon to be displayed.
  String iconUrl() {
    return iconUrl;
  }

  String iconName() {
    return iconName;
  }

  // Optional: Textual description that shows up as tooltip.
  String description() {
    return description;
  }

  // Optional: Foreground color.
  String foregroundColor() {
    return foregroundColor;
  }

  // Optional: Background color.
  String backgroundColor() {
    return backgroundColor;
  }

  public static Builder builder() {
    return new Builder()
        .setIconName("")
        .setIconUrl("")
        .setDescription("")
        .setForegroundColor("")
        .setBackgroundColor("");
  }

  /** Builder is the builder class for TestIntegration */
  public static class Builder {
    private String contactEmail;
    private String componentId;
    private String name;
    private String url;
    private String iconUrl;
    private String iconName;
    private String description;
    private String foregroundColor;
    private String backgroundColor;

    private Builder() {
    }

    /**
     * Sets the Contact Email value. The contact email is used for users to identify how to contact
     * the TestIntegration owner. This is optional.
     * @param email Email of the team responsible for this TestIntegration.
     * @return Builder
     */
    public Builder setContactEmail(String email) {
      this.contactEmail = email;
      return this;
    }

    /**
     * Sets the component ID value, used to identify the tool that this TestIntegration belongs to.
     * This is optional.
     * @param id ID of the component.
     * @return Builder
     */
    public Builder setComponentId(String id) {
      this.componentId = id;
      return this;
    }

    /**
     * Sets the name for the tool for this TestIntegration.
     * @param name Name of this TestIntegration.
     * @return Builder
     */
    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    /**
     * Sets the URL of this TestIntegration. It should be a FQDN, with optional url
     * encoded parameters.
     * @param url The location of the TestIntegration.
     * @return Builder
     */
    public Builder setUrl(String url) {
      this.url = url;
      return this;
    }

    /**
     * Sets the url of the icon. The icon should look good even if scaled down to 16x16.
     * This is optional; if not set, it will instead use the value passed to
     * {@link #setIconName(String)}.
     * @param  iconUrl Location of the icon.
     * @return Builder
     */
    public Builder setIconUrl(String iconUrl) {
      this.iconUrl = iconUrl;
      return this;
    }

    /**
     * Sets the name of the icon. This is optional; if not set it will instead use the value passed
     * to {@link #setIconUrl(String)}.
     *
     * @param iconName name of the icon.
     * @return Builder
     */
    public Builder setIconName(String iconName) {
      this.iconName = iconName;
      return this;
    }

    /**
     * Sets the description. The description is used to describe the TestIntegration object's
     * purpose. This is optional; if it isn't set, it will have a default value of {@code ""}.
     * @param  description The description for this TestIntegration.
     * @return Builder
     */
    public Builder setDescription(String description) {
      this.description = description;
      return this;
    }

    /**
     * Sets the foreground color of the TestIntegration link. This is optional; if it isn't set,
     * the link created will use the default foreground color per the tool's CSS.
     * @param foregroundColor The foreground color of the link, e.g. {@code "#000000"}.
     * @return Builder
     */
    public Builder setForegroundColor(String foregroundColor) {
      this.foregroundColor = foregroundColor;
      return this;
    }

    /**
     * Sets the background color of the TestIntegration link. This is optional; if it isn't set,
     * the link created will use the default background color per the tool's CSS.
     * @param backgroundColor The background color of the link, e.g. {@code "#ffffff"}.
     * @return Builder
     */
    public Builder setBackgroundColor(String backgroundColor) {
      this.backgroundColor = backgroundColor;
      return this;
    }

    /**
     * Builds a TestIntegration object.
     * @return Builder
     */
    public TestIntegration build() {
      return new TestIntegration(contactEmail, componentId, name, url, iconUrl, iconName,
          description, foregroundColor, backgroundColor);
    }
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
