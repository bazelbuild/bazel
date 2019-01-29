// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.FileType;

/**
 * Static helper class for creating and accessing Python provider information.
 *
 * <p>This class exposes a unified view over both the legacy and modern Python providers.
 */
public class PyProviderUtils {

  // Disable construction.
  private PyProviderUtils() {}

  /** Returns whether a given target has the {@code PyInfo} provider. */
  public static boolean hasModernProvider(TransitiveInfoCollection target) {
    return target.get(PyInfo.PROVIDER) != null;
  }

  /**
   * Returns the {@code PyInfo} provider from the given target info, or null if the provider is not
   * present.
   */
  public static PyInfo getModernProvider(TransitiveInfoCollection target) {
    return target.get(PyInfo.PROVIDER);
  }

  /** Returns whether a given target has the legacy "py" provider. */
  public static boolean hasLegacyProvider(TransitiveInfoCollection target) {
    return target.get(PyStructUtils.PROVIDER_NAME) != null;
  }

  /**
   * Returns the struct representing the legacy "py" provider, from the given target info.
   *
   * @throws EvalException if the provider does not exist or has the wrong type.
   */
  public static StructImpl getLegacyProvider(TransitiveInfoCollection target) throws EvalException {
    Object info = target.get(PyStructUtils.PROVIDER_NAME);
    if (info == null) {
      throw new EvalException(/*location=*/ null, "Target does not have 'py' provider");
    }
    return SkylarkType.cast(
        info,
        StructImpl.class,
        null,
        "'%s' provider should be a struct",
        PyStructUtils.PROVIDER_NAME);
  }

  /**
   * Returns the transitive sources of a given target.
   *
   * <p>If the target has a py provider, the value from that provider is used. Otherwise, we fall
   * back on collecting .py source files from the target's filesToBuild.
   *
   * @throws EvalException if the legacy struct provider is present but malformed
   */
  // TODO(bazel-team): Eliminate the fallback behavior by returning an appropriate py provider from
  // the relevant rules.
  public static NestedSet<Artifact> getTransitiveSources(TransitiveInfoCollection target)
      throws EvalException {
    if (hasModernProvider(target)) {
      return getModernProvider(target).getTransitiveSources().getSet(Artifact.class);
    } else if (hasLegacyProvider(target)) {
      return PyStructUtils.getTransitiveSources(getLegacyProvider(target));
    } else {
      NestedSet<Artifact> files = target.getProvider(FileProvider.class).getFilesToBuild();
      return NestedSetBuilder.<Artifact>compileOrder()
          .addAll(FileType.filter(files, PyRuleClasses.PYTHON_SOURCE))
          .build();
    }
  }

  /**
   * Returns whether a target uses shared libraries.
   *
   * <p>If the target has a py provider, the value from that provider is used. Otherwise, we fall
   * back on checking whether the target's filesToBuild contains a shared library file type (e.g., a
   * .so file).
   *
   * @throws EvalException if the legacy struct provider is present but malformed
   */
  public static boolean getUsesSharedLibraries(TransitiveInfoCollection target)
      throws EvalException {
    if (hasModernProvider(target)) {
      return getModernProvider(target).getUsesSharedLibraries();
    } else if (hasLegacyProvider(target)) {
      return PyStructUtils.getUsesSharedLibraries(getLegacyProvider(target));
    } else {
      NestedSet<Artifact> files = target.getProvider(FileProvider.class).getFilesToBuild();
      return FileType.contains(files, CppFileTypes.SHARED_LIBRARY);
    }
  }

  /**
   * Returns the transitive import paths of a target.
   *
   * <p>Imports are not currently propagated correctly (#7054). Currently the behavior is to return
   * the imports contained in the target's {@link PythonImportsProvider}, ignoring the py provider,
   * or no imports if there is no {@code PythonImportsProvider}. When #7054 is fixed, we'll instead
   * return the imports specified by the py provider, or those from {@code PythonImportsProvider} if
   * the py provider is not present, with an eventual goal of removing {@code
   * PythonImportsProvider}.
   */
  // TODO(#7054) Implement the above change.
  public static NestedSet<String> getImports(TransitiveInfoCollection target) throws EvalException {
    PythonImportsProvider importsProvider = target.getProvider(PythonImportsProvider.class);
    if (importsProvider != null) {
      return importsProvider.getTransitivePythonImports();
    } else {
      return NestedSetBuilder.emptySet(Order.COMPILE_ORDER);
    }
  }

  /**
   * Returns whether the target has transitive sources requiring Python 2.
   *
   * <p>If the target has a py provider, the value from that provider is used. Otherwise, we default
   * to false.
   */
  public static boolean getHasPy2OnlySources(TransitiveInfoCollection target) throws EvalException {
    if (hasModernProvider(target)) {
      return getModernProvider(target).getHasPy2OnlySources();
    } else if (hasLegacyProvider(target)) {
      return PyStructUtils.getHasPy2OnlySources(getLegacyProvider(target));
    } else {
      return false;
    }
  }

  /**
   * Returns whether the target has transitive sources requiring Python 3.
   *
   * <p>If the target has a py provider, the value from that provider is used. Otherwise, we default
   * to false.
   */
  public static boolean getHasPy3OnlySources(TransitiveInfoCollection target) throws EvalException {
    if (hasModernProvider(target)) {
      return getModernProvider(target).getHasPy3OnlySources();
    } else if (hasLegacyProvider(target)) {
      return PyStructUtils.getHasPy3OnlySources(getLegacyProvider(target));
    } else {
      return false;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  /** A builder to add both the legacy and modern providers to a configured target. */
  public static class Builder {
    private final PyInfo.Builder modernBuilder = PyInfo.builder();
    private final PyStructUtils.Builder legacyBuilder = PyStructUtils.builder();

    // Use the static builder() method instead.
    private Builder() {}

    public Builder setTransitiveSources(NestedSet<Artifact> transitiveSources) {
      modernBuilder.setTransitiveSources(transitiveSources);
      legacyBuilder.setTransitiveSources(transitiveSources);
      return this;
    }

    public Builder setUsesSharedLibraries(boolean usesSharedLibraries) {
      modernBuilder.setUsesSharedLibraries(usesSharedLibraries);
      legacyBuilder.setUsesSharedLibraries(usesSharedLibraries);
      return this;
    }

    public Builder setImports(NestedSet<String> imports) {
      modernBuilder.setImports(imports);
      legacyBuilder.setImports(imports);
      return this;
    }

    public Builder setHasPy2OnlySources(boolean hasPy2OnlySources) {
      modernBuilder.setHasPy2OnlySources(hasPy2OnlySources);
      legacyBuilder.setHasPy2OnlySources(hasPy2OnlySources);
      return this;
    }

    public Builder setHasPy3OnlySources(boolean hasPy3OnlySources) {
      modernBuilder.setHasPy3OnlySources(hasPy3OnlySources);
      legacyBuilder.setHasPy3OnlySources(hasPy3OnlySources);
      return this;
    }

    public RuleConfiguredTargetBuilder buildAndAddToTarget(
        RuleConfiguredTargetBuilder targetBuilder) {
      targetBuilder.addSkylarkTransitiveInfo(PyStructUtils.PROVIDER_NAME, legacyBuilder.build());
      targetBuilder.addNativeDeclaredProvider(modernBuilder.build());
      return targetBuilder;
    }
  }
}
