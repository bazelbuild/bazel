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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory.BuildInfoKey;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.Injectable;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * A value that represents something computed outside of the skyframe framework. These values are
 * "precomputed" from skyframe's perspective and so the graph needs to be prepopulated with them
 * (e.g. via injection).
 */
@AutoCodec
public class PrecomputedValue implements SkyValue {
  /**
   * An externally-injected precomputed value. Exists so that modules can inject precomputed values
   * into Skyframe's graph.
   *
   * @see com.google.devtools.build.lib.runtime.BlazeModule#getPrecomputedValues
   */
  public static final class Injected {
    private final Precomputed<?> precomputed;
    private final Supplier<? extends Object> supplier;

    private Injected(Precomputed<?> precomputed, Supplier<? extends Object> supplier) {
      this.precomputed = precomputed;
      this.supplier = supplier;
    }

    public void inject(Injectable injectable) {
      injectable.inject(precomputed.key, new PrecomputedValue(supplier.get()));
    }

    @Override
    public String toString() {
      return precomputed + ": " + supplier.get();
    }
  }

  public static <T> Injected injected(Precomputed<T> precomputed, Supplier<T> value) {
    return new Injected(precomputed, value);
  }

  public static <T> Injected injected(Precomputed<T> precomputed, T value) {
    return new Injected(precomputed, Suppliers.ofInstance(value));
  }

  public static final Precomputed<RuleVisibility> DEFAULT_VISIBILITY =
      new Precomputed<>(Key.create("default_visibility"));

  public static final Precomputed<StarlarkSemantics> STARLARK_SEMANTICS =
      new Precomputed<>(Key.create("skylark_semantics"));

  static final Precomputed<UUID> BUILD_ID =
      new Precomputed<>(Key.create("build_id"), /*shareable=*/ false);

  public static final Precomputed<Map<String, String>> ACTION_ENV =
      new Precomputed<>(Key.create("action_env"));

  public static final Precomputed<Map<String, String>> REPO_ENV =
      new Precomputed<>(Key.create("repo_env"));

  static final Precomputed<ImmutableList<ActionAnalysisMetadata>> COVERAGE_REPORT_KEY =
      new Precomputed<>(Key.create("coverage_report_actions"));

  public static final Precomputed<Map<BuildInfoKey, BuildInfoFactory>> BUILD_INFO_FACTORIES =
      new Precomputed<>(Key.create("build_info_factories"));

  public static final Precomputed<PathPackageLocator> PATH_PACKAGE_LOCATOR =
      new Precomputed<>(Key.create("path_package_locator"));

  public static final Precomputed<RemoteOutputsMode> REMOTE_OUTPUTS_MODE =
      new Precomputed<>(Key.create("remote_outputs_mode"));

  public static final Precomputed<Map<String, String>> REMOTE_DEFAULT_PLATFORM_PROPERTIES =
      new Precomputed<>(Key.create("remote_default_platform_properties"));

  public static final Precomputed<Boolean> REMOTE_EXECUTION_ENABLED =
      new Precomputed<>(Key.create("remote_execution_enabled"));

  private final Object value;

  @AutoCodec.Instantiator
  public PrecomputedValue(Object value) {
    this.value = Preconditions.checkNotNull(value);
  }

  /**
   * Returns the value of the variable.
   */
  public Object get() {
    return value;
  }

  @Override
  public int hashCode() {
    return value.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof PrecomputedValue)) {
      return false;
    }
    PrecomputedValue other = (PrecomputedValue) obj;
    return value.equals(other.value);
  }

  @Override
  public String toString() {
    return "<BuildVariable " + value + ">";
  }

  public static void dependOnBuildId(SkyFunction.Environment env) throws InterruptedException {
    BUILD_ID.get(env);
  }

  /**
   * A helper object corresponding to a variable in Skyframe.
   *
   * <p>Instances do not have internal state.
   */
  public static final class Precomputed<T> {
    private final Key key;
    private final boolean shareable;

    public Precomputed(Key key) {
      this(key, /*shareable=*/ true);
    }

    private Precomputed(Key key, boolean shareable) {
      this.key = key;
      this.shareable = shareable;
    }

    @VisibleForTesting
    public Key getKeyForTesting() {
      return key;
    }

    /**
     * Retrieves the value of this variable from Skyframe.
     *
     * <p>If the value was not set, an exception will be raised.
     */
    @Nullable
    @SuppressWarnings("unchecked")
    public T get(SkyFunction.Environment env) throws InterruptedException {
      PrecomputedValue value = (PrecomputedValue) env.getValue(key);
      if (value == null) {
        return null;
      }
      return (T) value.get();
    }

    /** Injects a new variable value. */
    public void set(Injectable injectable, T value) {
      injectable.inject(
          key, shareable ? new PrecomputedValue(value) : new UnshareablePrecomputedValue(value));
    }
  }

  /** An unshareable version of {@link PrecomputedValue}. */
  private static final class UnshareablePrecomputedValue extends PrecomputedValue {
    private UnshareablePrecomputedValue(Object value) {
      super(value);
    }

    @Override
    public boolean dataIsShareable() {
      return false;
    }
  }

  /**
   * {@link com/google/devtools/build/lib/skyframe/PrecomputedValue.java used only in javadoc:
   * com.google.devtools.build.skyframe.SkyKey} for {@code PrecomputedValue}.
   */
  @AutoCodec
  public static class Key extends AbstractSkyKey<String> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(String arg) {
      super(arg);
    }

    @AutoCodec.Instantiator
    public static Key create(String arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PRECOMPUTED;
    }
  }
}
