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
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;

/** Information about the system APIs for a Java compilation. */
@Immutable
public class BootClassPathInfo extends StarlarkInfoWrapper {

  private static final BootClassPathInfo EMPTY =
      new BootClassPathInfo(null) {
        @Override
        public NestedSet<Artifact> bootclasspath() {
          return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
        }

        @Override
        public NestedSet<Artifact> auxiliary() {
          return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
        }

        @Override
        public NestedSet<Artifact> systemInputs() {
          return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
        }

        @Override
        public Optional<PathFragment> systemPath() {
          return Optional.empty();
        }

        @Override
        public boolean isEmpty() {
          return true;
        }
      };

  public static BootClassPathInfo empty() {
    return EMPTY;
  }

  private BootClassPathInfo(StructImpl underlying) {
    super(underlying);
  }

  public static BootClassPathInfo wrap(Info info) throws RuleErrorException {
    return new BootClassPathInfo((StructImpl) info);
  }

  /** The jar files containing classes for system APIs, i.e. a Java <= 8 bootclasspath. */
  public NestedSet<Artifact> bootclasspath() throws RuleErrorException {
    return getUnderlyingNestedSet("bootclasspath", Artifact.class);
  }

  /**
   * The jar files containing extra classes for system APIs that should not be put in the system
   * image to support split-package compilation scenarios.
   */
  public NestedSet<Artifact> auxiliary() throws RuleErrorException {
    return getUnderlyingNestedSet("_auxiliary", Artifact.class);
  }

  /** Contents of the directory that is passed to the javac >= 9 {@code --system} flag. */
  public NestedSet<Artifact> systemInputs() throws RuleErrorException {
    return getUnderlyingNestedSet("_system_inputs", Artifact.class);
  }

  /** An argument to the javac >= 9 {@code --system} flag. */
  public Optional<PathFragment> systemPath() throws RuleErrorException {
    return Optional.ofNullable(getUnderlyingValue("_system_path", String.class))
        .map(PathFragment::create);
  }

  public boolean isEmpty() throws RuleErrorException {
    return bootclasspath().isEmpty()
        && auxiliary().isEmpty()
        && systemInputs().isEmpty()
        && systemPath().isEmpty();
  }
}
