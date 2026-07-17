// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.MaterializedDepsInfoApi;
import com.google.devtools.build.lib.util.Either;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;

/** The provider returned from materializer rules to materialize dependencies. */
@Immutable
public final class MaterializedDepsInfo extends NativeInfo implements MaterializedDepsInfoApi {

  public static final Provider PROVIDER = new Provider();

  /**
   * The dependencies to be materialized. These may be ConfiguredTarget or DormantDependency
   * objects.
   */
  private final ImmutableList<Either<ConfiguredTarget, DormantDependency>> deps;

  private MaterializedDepsInfo(ImmutableList<Either<ConfiguredTarget, DormantDependency>> deps) {
    this.deps = deps;
  }

  /**
   * The dependencies to be materialized. These may be ConfiguredTarget or DormantDependency
   * objects.
   */
  @Override
  public ImmutableList<Either<ConfiguredTarget, DormantDependency>> getDeps() {
    return deps;
  }

  @Override
  public Provider getProvider() {
    return PROVIDER;
  }

  /** Provider class for {@link MaterializedDepsInfo}. */
  public static class Provider extends BuiltinProvider<MaterializedDepsInfo>
      implements MaterializedDepsInfoApi.Provider {

    private Provider() {
      super(MaterializedDepsInfoApi.NAME, MaterializedDepsInfo.class);
    }

    @Override
    public MaterializedDepsInfoApi materializedDepsInfo(Sequence<?> dependencies)
        throws EvalException {

      ImmutableList.Builder<Either<ConfiguredTarget, DormantDependency>> depsBuilder =
          ImmutableList.builder();
      int index = 0;
      for (Object dependency : dependencies) {
        switch (dependency) {
          // Note that materializer rules can depend only on dependency_resolution_rule or
          // dormant dependencies, so the ConfiguredTargets here should be
          // dependency_resolution_rules.
          case ConfiguredTarget configuredTarget ->
              depsBuilder.add(Either.ofLeft(configuredTarget));
          case DormantDependency dormantDependency ->
              depsBuilder.add(Either.ofRight(dormantDependency));
          default ->
              throw Starlark.errorf(
                  "MaterializedDepsInfo dependencies must be Target objects (retrieved from"
                      + " ctx.attr) or DormantDependency objects (from attr.dormant_label() or"
                      + " attr.dormant_label_list() attributes), but got %s at index %s",
                  Starlark.type(dependency), index);
        }
        index++;
      }

      return new MaterializedDepsInfo(depsBuilder.build());
    }
  }
}
