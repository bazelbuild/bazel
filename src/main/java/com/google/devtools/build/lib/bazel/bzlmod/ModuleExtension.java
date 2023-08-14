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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Optional;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.syntax.Location;

/**
 * A module extension object, which can be used to perform arbitrary logic in order to create repos.
 */
@AutoValue
public abstract class ModuleExtension implements StarlarkValue {
  public abstract StarlarkCallable getImplementation();

  public abstract ImmutableMap<String, TagClass> getTagClasses();

  public abstract Optional<String> getDoc();

  /**
   * The .bzl file where the module extension object was originally defined.
   *
   * <p>Note that if the extension object was then loaded and re-exported by a different .bzl file
   * before being used in a MODULE.bazel file, the output of this function may differ from the
   * corresponding ModuleExtensionUsage#getExtensionBzlFile and ModuleExtensionId#getBzlFileLabel.
   */
  public abstract Label getDefiningBzlFileLabel();

  public abstract Location getLocation();

  public abstract ImmutableList<String> getEnvVariables();

  public static Builder builder() {
    return new AutoValue_ModuleExtension.Builder();
  }

  /** Builder for {@link ModuleExtension}. */
  @AutoValue.Builder
  public abstract static class Builder {
    public abstract Builder setDoc(Optional<String> value);

    public abstract Builder setDefiningBzlFileLabel(Label value);

    public abstract Builder setLocation(Location value);

    public abstract Builder setImplementation(StarlarkCallable value);

    public abstract Builder setTagClasses(ImmutableMap<String, TagClass> value);

    public abstract Builder setEnvVariables(ImmutableList<String> value);

    public abstract ModuleExtension build();
  }
}
