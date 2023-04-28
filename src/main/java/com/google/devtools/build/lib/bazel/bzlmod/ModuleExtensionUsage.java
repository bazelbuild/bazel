// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import net.starlark.java.syntax.Location;

/**
 * Represents one usage of a module extension in one MODULE.bazel file. This class records all the
 * information pertinent to the proxy object returned from the {@code use_extension} call.
 */
@AutoValue
@GenerateTypeAdapter
public abstract class ModuleExtensionUsage {
  /** An unresolved label pointing to the Starlark file where the module extension is defined. */
  public abstract String getExtensionBzlFile();

  /** The name of the extension. */
  public abstract String getExtensionName();

  /** The module that contains this particular extension usage. */
  public abstract ModuleKey getUsingModule();

  /**
   * The location where this proxy object was created (by the {@code use_extension} call). Note that
   * if there were multiple {@code use_extension} calls on same extension, then this only stores the
   * location of the first one.
   */
  public abstract Location getLocation();

  /**
   * All the repos imported from this module extension into the scope of the current module. The key
   * is the local repo name (in the scope of the current module), and the value is the name exported
   * by the module extension.
   */
  public abstract ImmutableBiMap<String, String> getImports();

  /**
   * The repo names as exported by the module extension that were imported using a proxy marked as a
   * dev dependency.
   */
  public abstract ImmutableSet<String> getDevImports();

  /** All the tags specified by this module for this extension. */
  public abstract ImmutableList<Tag> getTags();

  public static Builder builder() {
    return new AutoValue_ModuleExtensionUsage.Builder();
  }

  /** Builder for {@link ModuleExtensionUsage}. */
  @AutoValue.Builder
  public abstract static class Builder {

    public abstract Builder setExtensionBzlFile(String value);

    public abstract Builder setExtensionName(String value);

    public abstract Builder setUsingModule(ModuleKey value);

    public abstract Builder setLocation(Location value);

    public abstract Builder setImports(ImmutableBiMap<String, String> value);

    public abstract Builder setDevImports(ImmutableSet<String> value);

    public abstract Builder setTags(ImmutableList<Tag> value);

    abstract ImmutableList.Builder<Tag> tagsBuilder();

    @CanIgnoreReturnValue
    public ModuleExtensionUsage.Builder addTag(Tag value) {
      tagsBuilder().add(value);
      return this;
    }

    public abstract ModuleExtensionUsage build();
  }
}
