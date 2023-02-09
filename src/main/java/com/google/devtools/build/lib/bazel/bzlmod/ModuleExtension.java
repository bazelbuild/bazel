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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.syntax.Location;

/**
 * A module extension object, which can be used to perform arbitrary logic in order to create repos
 * or register toolchains and execution platforms.
 */
@AutoValue
public abstract class ModuleExtension {
  public abstract String getName();

  public abstract StarlarkCallable getImplementation();

  public abstract ImmutableMap<String, TagClass> getTagClasses();

  public abstract String getDoc();

  public abstract Label getDefinitionEnvironmentLabel();

  public abstract Location getLocation();

  public static Builder builder() {
    return new AutoValue_ModuleExtension.Builder();
  }

  /** Builder for {@link ModuleExtension}. */
  @AutoValue.Builder
  public abstract static class Builder {

    public abstract Builder setDoc(String value);

    public abstract Builder setDefinitionEnvironmentLabel(Label value);

    public abstract Builder setLocation(Location value);

    public abstract Builder setName(String value);

    public abstract Builder setImplementation(StarlarkCallable value);

    public abstract Builder setTagClasses(ImmutableMap<String, TagClass> value);

    public abstract ModuleExtension build();
  }

  /**
   * A {@link ModuleExtension} exposed to Starlark. We can't use {@link ModuleExtension} directly
   * because the name isn't known until the object is exported, so this class holds a builder until
   * it's exported, at which point it sets the name and builds the underlying {@link
   * ModuleExtension}.
   */
  @StarlarkBuiltin(
      name = "module_extension",
      category = DocCategory.BUILTIN,
      doc = "A module extension declared using the <code>module_extension</code> function.")
  public static class InStarlark implements StarlarkExportable {
    private final Builder builder;
    @Nullable private ModuleExtension built;

    public InStarlark() {
      builder = builder();
      built = null;
    }

    public Builder getBuilder() {
      return builder;
    }

    @Override
    public boolean isExported() {
      return built != null;
    }

    @Override
    public void export(EventHandler handler, Label extensionLabel, String exportedName) {
      built = builder.setName(exportedName).build();
    }

    /** Throws {@link IllegalStateException} if this is not exported yet. */
    public ModuleExtension get() {
      Preconditions.checkState(isExported(), "the module extension was never exported");
      return built;
    }
  }
}
