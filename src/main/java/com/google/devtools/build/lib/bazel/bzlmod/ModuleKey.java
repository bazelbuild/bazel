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
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import java.util.Comparator;
import java.util.Optional;
import javax.annotation.Nullable;

/** A module name, version pair that identifies a module in the external dependency graph. */
public class ModuleKey {

  /**
   * A mapping from module name to repository name for certain special "well-known" modules.
   *
   * <p>The repository name of certain modules are required to be exact strings (instead of the
   * normal format seen in {@link #getCanonicalRepoName()}) due to backwards compatibility reasons.
   * For example, bazel_tools must be known as "@bazel_tools" for WORKSPACE repos to work correctly.
   *
   * <p>TODO(wyv): After we make all flag values go through repo mapping, we can remove the concept
   * of well-known modules altogether.
   */
  private static final ImmutableMap<String, RepositoryName> WELL_KNOWN_MODULES =
      ImmutableMap.of(
          "bazel_tools",
          RepositoryName.BAZEL_TOOLS,
          "local_config_platform",
          RepositoryName.createUnvalidated("local_config_platform"));

  public static final ModuleKey ROOT = create("", Version.EMPTY);

  public static final Comparator<ModuleKey> LEXICOGRAPHIC_COMPARATOR =
      Comparator.comparing(ModuleKey::getName).thenComparing(ModuleKey::getVersion);

  public static ModuleKey create(String name, Version version) {
    return new ModuleKey(name, version, null);
  }

  public static ModuleKey create(String name, Version version, boolean hasGloballyUniqueVersion) {
    return new ModuleKey(name, version, hasGloballyUniqueVersion);
  }

  private final String name;
  private final Version version;
  private final Boolean hasGloballyUniqueVersion;

  private ModuleKey(String name, Version version, @Nullable Boolean hasGloballyUniqueVersion) {
    this.name = name;
    this.version = version;
    this.hasGloballyUniqueVersion = hasGloballyUniqueVersion;
  }

  /** The name of the module. Must be empty for the root module. */
  public String getName() {
    return name;
  }

  /** The version of the module. Must be empty iff the module has a {@link NonRegistryOverride}. */
  public Version getVersion() {
    return version;
  }

  public Optional<Boolean> hasGloballyUniqueVersion() {
    return Optional.ofNullable(hasGloballyUniqueVersion);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(getName(), getVersion());
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (o instanceof ModuleKey) {
      ModuleKey that = (ModuleKey) o;
      return this.name.equals(that.getName())
          && this.version.equals(that.getVersion());
    }
    return false;
  }

  @Override
  public final String toString() {
    if (this.equals(ROOT)) {
      return "<root>";
    }
    return getName() + "@" + (getVersion().isEmpty() ? "_" : getVersion().toString());
  }

  /** Returns the canonical name of the repo backing this module. */
  public RepositoryName getCanonicalRepoName() {
    if (WELL_KNOWN_MODULES.containsKey(getName())) {
      return WELL_KNOWN_MODULES.get(getName());
    }
    if (ROOT.equals(this)) {
      return RepositoryName.MAIN;
    }
    if (getVersion().isEmpty() || !hasGloballyUniqueVersion().get()) {
      return RepositoryName.createUnvalidated(
          String.format("%s~%s", getName(), getVersion().isEmpty() ? "override" : getVersion()));
    } else {
      return RepositoryName.createUnvalidated(getName());
    }
  }
}
