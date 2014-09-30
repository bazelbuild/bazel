// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.ActionGraphVisitor;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Dumper;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.common.options.EnumConverter;

import java.io.PrintStream;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.SortedSet;

/**
 * Dumps the targets of a build.
 */
class TargetsDumper implements Dumper {

  /** Enumeration of the supported output formats. */
  public enum FormatType {

    /** Output the labels of the input files in alphabetical order. */
    INPUTS {
      @Override
      protected String asString(Target target) {
        return target instanceof InputFile ? target.getLabel().toString() + "\n" : "";
      }
    },
    /** Output the labels of the traversed rules in alphabetical order. */
    RULES {
      @Override
      protected String asString(Target target) {
        return !(target instanceof InputFile) ? target.getLabel().toString() + "\n" : "";
      }
    },
    /**
     * Output the labels of the traversed rules and the input files in
     * alphabetical order.
     */
    ALL {
      @Override
      protected String asString(Target target) {
        return target.getLabel().toString() + "\n";
      }
    },
    /**
     * Output the packages of the traversed rules and the input files in
     * alphabetical order.
     */
    PACKAGES {
      @Override
      protected String asString(Target target) {
        return target.getLabel().getPackageName() + "\n";
      }
    };

    /** Converts option to {@link FormatType}. */
    public static class Converter extends EnumConverter<FormatType> {
      public Converter() {
        super(FormatType.class, "format type");
      }
    }

    /**
     * Prints the given {@link Target}s to the given {@link PrintStream}
     *
     * @param targets the targets to print
     * @param o the stream to print the targets to
     */
    private void output(Set<Target> targets, PrintStream o) {
      SortedSet<String> strings = Sets.newTreeSet();
      for (Target target : targets) {
        strings.add(asString(target));
      }
      for (String string : strings) {
        o.print(string);
      }
    }

    /**
     * Converts an {@link InputFile} to a {@link String} including newline.
     *
     * @param target the target to convert
     * @return the string representation of the input file
     */
    protected abstract String asString(Target target);
  }

  /** The Artifacts from which the traversal of the action graph should start. */
  private final Collection<Artifact> roots;

  /** The action graph. */
  private final ActionGraph actionGraph;

  /** The package cache. */
  private final LoadedPackageProvider packageCache;

  /** The format of the output. */
  private final FormatType format;

  /** Indicates whether host dependencies are dumped. */
  private final boolean dumpHostDeps;

  /**
   * @param roots the Artifacts from which the traversal of the action graph
   *        should start
   * @param actionGraph
   * @param format the format of the output
   * @param dumpHostDeps if true, host dependencies are dumped
   */
  public TargetsDumper(Collection<Artifact> roots, ActionGraph actionGraph,
      LoadedPackageProvider packageCache, FormatType format, boolean dumpHostDeps) {
    this.dumpHostDeps = dumpHostDeps;
    this.roots = Preconditions.checkNotNull(roots);
    this.actionGraph = Preconditions.checkNotNull(actionGraph);
    this.format = Preconditions.checkNotNull(format);
    this.packageCache = Preconditions.checkNotNull(packageCache);
  }

  @Override
  public void dump(PrintStream out) {
    Preconditions.checkNotNull(out);

    CollectInputFilesVisitor visitor = new CollectInputFilesVisitor();
    visitor.visitRootArtifacts(roots);
    try {
      format.output(visitor.getTargets(), out);
    } finally {
      out.flush();
    }
  }

  @Override
  public String getFileName() {
    return "BlazeTargets.txt";
  }

  @Override
  public String getName() {
    return "Targets";
  }

  /** Visits the action graph and collects all {@link InputFile}s. */
  private class CollectInputFilesVisitor extends ActionGraphVisitor {

    private final Set<Target> targets = new HashSet<>();

    public CollectInputFilesVisitor() {
      super(actionGraph);
    }

    @Override
    protected void visitArtifact(Artifact artifact) {
      if (artifact.getOwner() != null) {
        try {
          targets.add(packageCache.getLoadedTarget(artifact.getOwner()));
        } catch (NoSuchThingException e) {
          throw new IllegalStateException();
        }
      }
    }

    @Override
    protected boolean shouldVisit(Action action) {
      if (dumpHostDeps || !"host".equals(action.getOwner().getConfigurationName())) {
        return true;
      }
      return false;
    }

    @Override
    protected void visitAction(Action action) {
      if (action.getOwner() != null && action.getOwner().getLabel() != null) {
        try {
          targets.add(packageCache.getLoadedTarget(action.getOwner().getLabel()));
        } catch (NoSuchThingException e) {
          throw new IllegalStateException();
        }
      }
    }

    public void visitRootArtifacts(Collection<Artifact> artifacts) {
      visitWhiteNodes(artifacts);
    }

    public Set<Target> getTargets() {
      return targets;
    }
  }
}
