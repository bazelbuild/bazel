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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Function;
import com.google.common.base.Objects;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Runfiles a target contributes to targets that depend on it.
 *
 * <p>The set of runfiles contributed can be different if the dependency is through a <code>data
 * </code> attribute (note that this is just a rough approximation of the reality -- rule
 * implementations are free to request the data runfiles at any time)
 */
@Immutable
public final class RunfilesProvider implements TransitiveInfoProvider {

  /**
   * Holds a {@link RepositoryName} together with the corresponding {@link RepositoryMapping}.
   *
   * <p>Instances of this class compare equal iff the <code>repositoryName</code> members are
   * equal. As a result, a {@link Set<RepositoryNameAndMapping>} will behave like an
   * {@link Map#entrySet} of a {@link Map Map<RepositoryName, RepositoryMapping>}.
   */
  public static final class RepositoryNameAndMapping {

    private final RepositoryName repositoryName;
    private final RepositoryMapping repositoryMapping;

    public RepositoryNameAndMapping(RepositoryName repositoryName,
        RepositoryMapping repositoryMapping) {
      this.repositoryName = repositoryName;
      this.repositoryMapping = repositoryMapping;
    }

    public RepositoryName getRepositoryName() {
      return repositoryName;
    }

    public RepositoryMapping getRepositoryMapping() {
      return repositoryMapping;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      RepositoryNameAndMapping that = (RepositoryNameAndMapping) o;
      return Objects.equal(repositoryName, that.repositoryName);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(repositoryName);
    }
  }

  private final Runfiles defaultRunfiles;
  private final Runfiles dataRunfiles;
  private final NestedSet<RepositoryNameAndMapping> runfilesLibraryUsers;

  private RunfilesProvider(Runfiles defaultRunfiles, Runfiles dataRunfiles,
      NestedSet<RepositoryNameAndMapping> runfilesLibraryUsers) {
    this.defaultRunfiles = defaultRunfiles;
    this.dataRunfiles = dataRunfiles;
    this.runfilesLibraryUsers = runfilesLibraryUsers;
  }

  public Runfiles getDefaultRunfiles() {
    return defaultRunfiles;
  }

  public Runfiles getDataRunfiles() {
    return dataRunfiles;
  }

  public NestedSet<RepositoryNameAndMapping> getRunfilesLibraryUsers() {
    return runfilesLibraryUsers;
  }

  /**
   * Returns a function that gets the default runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> DEFAULT_RUNFILES =
      new Function<TransitiveInfoCollection, Runfiles>() {
        @Override
        public Runfiles apply(TransitiveInfoCollection input) {
          RunfilesProvider provider = input.getProvider(RunfilesProvider.class);
          if (provider != null) {
            return provider.getDefaultRunfiles();
          }

          return Runfiles.EMPTY;
        }
      };

  /**
   * Returns a function that gets the data runfiles from a {@link TransitiveInfoCollection} or the
   * empty runfiles instance if it does not contain that provider.
   *
   * <p>These are usually used if the target is depended on through a {@code data} attribute.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> DATA_RUNFILES =
      new Function<TransitiveInfoCollection, Runfiles>() {
        @Override
        public Runfiles apply(TransitiveInfoCollection input) {
          RunfilesProvider provider = input.getProvider(RunfilesProvider.class);
          if (provider != null) {
            return provider.getDataRunfiles();
          }

          return Runfiles.EMPTY;
        }
      };

  public static RunfilesProvider simple(RuleContext ruleContext, Runfiles defaultRunfiles) {
    return new RunfilesProvider(defaultRunfiles, defaultRunfiles,
        collectRunfilesLibraryUsers(ruleContext));
  }

  public static RunfilesProvider withData(
      RuleContext ruleContext, Runfiles defaultRunfiles, Runfiles dataRunfiles) {
    return new RunfilesProvider(defaultRunfiles, dataRunfiles,
        collectRunfilesLibraryUsers(ruleContext));
  }

  public static final RunfilesProvider EMPTY = new RunfilesProvider(
      Runfiles.EMPTY, Runfiles.EMPTY, NestedSetBuilder.emptySet(Order.COMPILE_ORDER));

  /**
   * Collects the runfiles library users of all non-tool dependencies and adds the current
   * repository if a runfiles library (marked with {@link RunfilesLibraryInfo}) is among these
   * dependencies.
   */
  private static NestedSet<RepositoryNameAndMapping> collectRunfilesLibraryUsers(
      RuleContext ruleContext) {
    NestedSetBuilder<RepositoryNameAndMapping> users = NestedSetBuilder.compileOrder();
    for (TransitiveInfoCollection dep : getAllNonToolPrerequisites(ruleContext)) {
      RunfilesProvider provider = dep.getProvider(RunfilesProvider.class);
      if (provider != null) {
        users.addTransitive(provider.getRunfilesLibraryUsers());
      }
      if (dep.get(RunfilesLibraryInfo.PROVIDER) != null) {
        users.add(new RepositoryNameAndMapping(ruleContext.getRepository(),
            ruleContext.getRule().getPackage().getRepositoryMapping()));
      }
    }
    return users.build();
  }

  private static Iterable<TransitiveInfoCollection> getAllNonToolPrerequisites(
      RuleContext ruleContext) {
    List<TransitiveInfoCollection> prerequisites = new ArrayList<>();
    for (Attribute attribute : ruleContext.getRule().getAttributes()) {
      if (attribute.getType().getLabelClass() != LabelClass.DEPENDENCY
          || attribute.isToolDependency()) {
        continue;
      }
      prerequisites.addAll(ruleContext.getPrerequisites(attribute.getName()));
    }
    return prerequisites;
  }
}
