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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import net.starlark.java.syntax.Location;

/**
 * Represents a module extension tag, which is a piece of data following a specified attribute
 * schema that can be consumed by a module extension implementation function. The attribute schema
 * is defined by the {@link TagClass}, and checked at module extension resolution time (i.e.
 * <em>not</em> when the tag is created, which is during module discovery).
 */
@AutoValue
@GenerateTypeAdapter
public abstract class Tag {

  public abstract String getTagName();

  /** All keyword arguments supplied to the tag instance. */
  public abstract AttributeValues getAttributeValues();

  /** Whether this tag was created using a proxy created with dev_dependency = True. */
  public abstract boolean isDevDependency();

  /** The source location in the module file where this tag was created. */
  public abstract Location getLocation();

  public static Builder builder() {
    return new AutoValue_Tag.Builder();
  }

  /** Builder for {@link Tag}. */
  @AutoValue.Builder
  public abstract static class Builder {

    public abstract Builder setTagName(String value);

    public abstract Builder setAttributeValues(AttributeValues value);

    public abstract Builder setDevDependency(boolean value);

    public abstract Builder setLocation(Location value);

    public abstract Tag build();
  }
}
