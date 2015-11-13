// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Objects;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

import javax.annotation.Nullable;

/**
 * An aspect in the context of the Skyframe graph.
 */
public final class AspectValue extends ActionLookupValue {

  /**
   * A base class for keys that have AspectValue as a Sky value.
   */
  public abstract static class AspectValueKey extends ActionLookupKey {

    public abstract String getDescription();
  }

  /**
   * A base class for a key representing an aspect applied to a particular target.
   */
  public static final class AspectKey extends AspectValueKey {
    private final Label label;
    private final BuildConfiguration configuration;
    private final Aspect aspect;

    protected AspectKey(
        Label label,
        BuildConfiguration configuration,
        AspectClass aspectClass,
        AspectParameters parameters) {
      this.label = label;
      this.configuration = configuration;
      this.aspect = new Aspect(aspectClass, parameters);
    }

    @Override
    SkyFunctionName getType() {
      return SkyFunctions.ASPECT;
    }


    @Override
    public Label getLabel() {
      return label;
    }

    public AspectClass getAspectClass() {
      return aspect.getAspectClass();
    }

    @Nullable
    public AspectParameters getParameters() {
      return aspect.getParameters();
    }

    public Aspect getAspect() {
      return aspect;
    }

    @Override
    public String getDescription() {
      return String.format("%s of %s", aspect.getAspectClass().getName(), getLabel());
    }

    public BuildConfiguration getConfiguration() {
      return configuration;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(label, configuration, aspect);
    }

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }

      if (!(other instanceof AspectKey)) {
        return false;
      }

      AspectKey that = (AspectKey) other;
      return Objects.equal(label, that.label)
          && Objects.equal(configuration, that.configuration)
          && Objects.equal(aspect, that.aspect);
    }

    @Override
    public String toString() {
      return label
          + "#"
          + aspect.getAspectClass().getName()
          + " "
          + (configuration == null ? "null" : configuration.checksum())
          + " "
          + aspect.getParameters();
    }
  }

  /**
   * The key for a skylark aspect.
   */
  public static class SkylarkAspectLoadingKey extends AspectValueKey {

    private final Label targetLabel;
    private final BuildConfiguration targetConfiguration;
    private final PathFragment extensionFile;
    private final String skylarkValueName;

    private SkylarkAspectLoadingKey(
        Label targetLabel,
        BuildConfiguration targetConfiguration,
        PathFragment extensionFile,
        String skylarkFunctionName) {
      this.targetLabel = targetLabel;
      this.targetConfiguration = targetConfiguration;

      this.extensionFile = extensionFile;
      this.skylarkValueName = skylarkFunctionName;
    }

    @Override
    SkyFunctionName getType() {
      return SkyFunctions.LOAD_SKYLARK_ASPECT;
    }

    public PathFragment getExtensionFile() {
      return extensionFile;
    }

    public String getSkylarkValueName() {
      return skylarkValueName;
    }

    public Label getTargetLabel() {
      return targetLabel;
    }

    public BuildConfiguration getTargetConfiguration() {
      return targetConfiguration;
    }

    @Override
    public String getDescription() {
      // Skylark aspects are referred to on command line with <file>%<value ame>
      return String.format("%s%%%s of %s", extensionFile.toString(), skylarkValueName, targetLabel);
    }
  }


  private final Label label;
  private final Location location;
  private final AspectKey key;
  private final ConfiguredAspect configuredAspect;
  private final NestedSet<Package> transitivePackages;

  public AspectValue(
      AspectKey key,
      Label label,
      Location location,
      ConfiguredAspect configuredAspect,
      Iterable<Action> actions,
      NestedSet<Package> transitivePackages) {
    super(actions);
    this.location = location;
    this.label = label;
    this.key = key;
    this.configuredAspect = configuredAspect;
    this.transitivePackages = transitivePackages;
  }

  public ConfiguredAspect getConfiguredAspect() {
    return configuredAspect;
  }

  public Label getLabel() {
    return label;
  }

  public Location getLocation() {
    return location;
  }

  public AspectKey getKey() {
    return key;
  }

  public NestedSet<Package> getTransitivePackages() {
    return transitivePackages;
  }

  public static SkyKey key(
      Label label,
      BuildConfiguration configuration,
      AspectClass aspectFactory,
      AspectParameters additionalConfiguration) {
    return new SkyKey(
        SkyFunctions.ASPECT,
        new AspectKey(label, configuration, aspectFactory, additionalConfiguration));
  }

  public static SkyKey key(AspectValueKey aspectKey) {
    return new SkyKey(aspectKey.getType(), aspectKey);
  }

  public static AspectKey createAspectKey(
      Label label, BuildConfiguration configuration, AspectClass aspectFactory) {
    return new AspectKey(label, configuration, aspectFactory, AspectParameters.EMPTY);
  }

  public static SkylarkAspectLoadingKey createSkylarkAspectKey(
      Label targetLabel,
      BuildConfiguration targetConfiguration,
      PathFragment skylarkFile,
      String skylarkExportName) {
    return new SkylarkAspectLoadingKey(
        targetLabel, targetConfiguration, skylarkFile, skylarkExportName);
  }
}
