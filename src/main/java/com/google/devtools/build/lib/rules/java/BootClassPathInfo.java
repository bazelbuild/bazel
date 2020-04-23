// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;

/** Information about the system APIs for a Java compilation. */
@AutoCodec
@Immutable
public class BootClassPathInfo extends NativeInfo implements StarlarkValue {

  public static final String SKYLARK_NAME = "BootClassPathInfo";

  /** Provider singleton constant. */
  public static final BuiltinProvider<BootClassPathInfo> PROVIDER = new Provider();

  /** Provider class for {@link BootClassPathInfo} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  public static class Provider extends BuiltinProvider<BootClassPathInfo> implements ProviderApi {
    private Provider() {
      super(SKYLARK_NAME, BootClassPathInfo.class);
    }

    @SkylarkCallable(
        name = "BootClassPathInfo",
        doc = "The <code>BootClassPathInfo</code> constructor.",
        documented = false,
        parameters = {
          @Param(
              name = "bootclasspath",
              positional = false,
              named = true,
              type = Sequence.class,
              defaultValue = "[]"),
          @Param(
              name = "auxiliary",
              positional = false,
              named = true,
              type = Sequence.class,
              defaultValue = "[]"),
          @Param(
              name = "system",
              positional = false,
              named = true,
              type = FileApi.class,
              noneable = true,
              defaultValue = "None"),
        },
        selfCall = true,
        useStarlarkThread = true)
    public BootClassPathInfo bootClassPathInfo(
        Sequence<?> bootClassPathList,
        Sequence<?> auxiliaryList,
        Object systemOrNone,
        StarlarkThread thread)
        throws EvalException {
      return new BootClassPathInfo(
          getBootClassPath(bootClassPathList),
          getAuxiliary(auxiliaryList),
          getSystem(systemOrNone),
          thread.getCallerLocation());
    }

    private static NestedSet<Artifact> getBootClassPath(Sequence<?> bootClassPathList)
        throws EvalException {
      return NestedSetBuilder.wrap(
          Order.STABLE_ORDER, Sequence.cast(bootClassPathList, Artifact.class, "bootclasspath"));
    }

    private static NestedSet<Artifact> getAuxiliary(Sequence<?> auxiliaryList)
        throws EvalException {
      return NestedSetBuilder.wrap(
          Order.STABLE_ORDER, Sequence.cast(auxiliaryList, Artifact.class, "auxiliary"));
    }

    private static Artifact getSystem(Object systemOrNone) throws EvalException {
      if (systemOrNone == Starlark.NONE) {
        return null;
      }
      if (systemOrNone instanceof Artifact) {
        return (Artifact) systemOrNone;
      }
      throw Starlark.errorf("for system, got %s, want File or None", Starlark.type(systemOrNone));
    }
  }

  private final NestedSet<Artifact> bootclasspath;
  private final NestedSet<Artifact> auxiliary;
  @Nullable private final Artifact system;

  @VisibleForSerialization
  @AutoCodec.Instantiator
  public BootClassPathInfo(
      NestedSet<Artifact> bootclasspath,
      NestedSet<Artifact> auxiliary,
      Artifact system,
      Location location) {
    super(PROVIDER, location);
    this.bootclasspath = bootclasspath;
    this.auxiliary = auxiliary;
    this.system = system;
  }

  public static BootClassPathInfo create(NestedSet<Artifact> bootclasspath) {
    return new BootClassPathInfo(
        bootclasspath, NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER), null, null);
  }

  public static BootClassPathInfo empty() {
    return new BootClassPathInfo(
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER),
        null,
        null);
  }

  /** The jar files containing classes for system APIs, i.e. a Java <= 8 bootclasspath. */
  public NestedSet<Artifact> bootclasspath() {
    return bootclasspath;
  }

  /**
   * The jar files containing extra classes for system APIs that should not be put in the system
   * image to support split-package compilation scenarios.
   */
  public NestedSet<Artifact> auxiliary() {
    return auxiliary;
  }

  /** An argument to the javac >= 9 {@code --system} flag. */
  @Nullable
  public Artifact system() {
    return system;
  }

  public boolean isEmpty() {
    return bootclasspath.isEmpty();
  }
}
