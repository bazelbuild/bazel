// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Objects;

/**
 * {@link AspectClass} for aspects defined in Skylark.
 */
@Immutable
public final class SkylarkAspectClass implements AspectClass {

  private final Label extensionLabel;
  private final String exportedName;
  private final ImmutableList<String> attrAspects;
  private final ImmutableList<Attribute> attributes;
  private final ImmutableSet<String> configurationFragments;
  private final ImmutableSet<String> hostConfigurationFragments;

  private final AspectDefinition aspectDefinition;
  private final int hashCode;

  /** Builder class for {@link com.google.devtools.build.lib.packages.SkylarkAspectClass} */
  public static class Builder {
    private Label extensionLabel;
    private String exportedName;
    private ImmutableList.Builder<String> attrAspects = ImmutableList.builder();
    private ImmutableList.Builder<Attribute> attributes = ImmutableList.builder();
    private ImmutableSet.Builder<String> configurationFragments = ImmutableSet.builder();
    private ImmutableSet.Builder<String> hostConfigurationFragments = ImmutableSet.builder();;

    public Builder setExtensionLabel(Label extensionLabel) {
      this.extensionLabel = extensionLabel;
      return this;
    }

    public Builder setExportedName(String exportedName) {
      this.exportedName = exportedName;
      return this;
    }

    public Builder addAttrAspects(Iterable<String> attrAspects) {
      this.attrAspects.addAll(attrAspects);
      return this;
    }

    public Builder addAttribute(Attribute attribute) {
      this.attributes.add(attribute);
      return this;
    }

    public Builder addConfigurationFragments(Iterable<String> configurationFragments) {
      this.configurationFragments.addAll(configurationFragments);
      return this;
    }

    public Builder addHostConfigurationFragments(Iterable<String> hostConfigurationFragments) {
      this.hostConfigurationFragments.addAll(hostConfigurationFragments);
      return this;
    }

    public SkylarkAspectClass build() {
      return new SkylarkAspectClass(
          extensionLabel,
          exportedName,
          attrAspects.build(),
          attributes.build(),
          configurationFragments.build(),
          hostConfigurationFragments.build());
    }
  }

  private SkylarkAspectClass(
      Label extensionLabel,
      String exportedName,
      ImmutableList<String> attrAspects,
      ImmutableList<Attribute> attributes,
      ImmutableSet<String> configurationFragments,
      ImmutableSet<String> hostConfigurationFragments) {
    this.extensionLabel = Preconditions.checkNotNull(extensionLabel);
    this.exportedName = Preconditions.checkNotNull(exportedName);

    this.attrAspects = attrAspects;
    this.attributes = attributes;
    this.configurationFragments = configurationFragments;
    this.hostConfigurationFragments = hostConfigurationFragments;

    // Cache hash code.
    this.hashCode = Objects.hash(
        this.extensionLabel,
        this.exportedName,
        this.attrAspects,
        this.attributes,
        this.configurationFragments,
        this.hostConfigurationFragments);

    // Cache aspect definition
    this.aspectDefinition = buildDefinition();
  }

  private AspectDefinition buildDefinition() {
    AspectDefinition.Builder builder = new AspectDefinition.Builder(getName());
    for (String attrAspect : this.attrAspects) {
      builder.attributeAspect(attrAspect, this);
    }
    for (Attribute attribute : this.attributes) {
      builder.add(attribute);
    }
    builder.requiresConfigurationFragmentsBySkylarkModuleName(configurationFragments);
    builder.requiresHostConfigurationFragmentsBySkylarkModuleName(hostConfigurationFragments);
    return builder.build();
  }

  public final Label getExtensionLabel() {
    return extensionLabel;
  }

  public final String getExportedName() {
    return exportedName;
  }

  public ImmutableList<String> getAttrAspects() {
    return attrAspects;
  }

  public ImmutableList<Attribute> getAttributes() {
    return attributes;
  }

  public ImmutableSet<String> getConfigurationFragments() {
    return configurationFragments;
  }

  public ImmutableSet<String> getHostConfigurationFragments() {
    return hostConfigurationFragments;
  }

  @Override
  public final AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return aspectDefinition;
  }

  @Override
  public final String getName() {
    return getExtensionLabel() + "%" + getExportedName();
  }

  @Override
  public final boolean equals(Object o) {
    if (this == o) {
      return true;
    }

    if (!(o instanceof SkylarkAspectClass)) {
      return false;
    }

    SkylarkAspectClass that = (SkylarkAspectClass) o;

    return hashCode == that.hashCode
        && getExtensionLabel().equals(that.getExtensionLabel())
        && getExportedName().equals(that.getExportedName())
        && attrAspects.equals(that.attrAspects)
        && attributes.equals(that.attributes)
        && configurationFragments.equals(that.configurationFragments)
        && hostConfigurationFragments.equals(that.hostConfigurationFragments);
  }

  @Override
  public final int hashCode() {
    return hashCode;
  }
}
