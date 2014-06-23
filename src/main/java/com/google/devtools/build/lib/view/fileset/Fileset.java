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

package com.google.devtools.build.lib.view.fileset;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.view.AnalysisUtils;
import com.google.devtools.build.lib.view.FileProvider;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.GenericRuleConfiguredTargetBuilder.StatelessRunfilesProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.ConfiguredFilesetEntry;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesCollector;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.SymlinkTreeAction;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.Util;
import com.google.devtools.build.lib.view.actions.SymlinkAction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.fileset.FilesetUtil.SubpackageMode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadPoolExecutor;

import javax.annotation.Nullable;

/**
 * ConfiguredTarget for "Fileset".
 *
 * <p>Fileset rule provides a means to map (via symlink) input and generated files
 * into the output directory.
 *
 * <p>There are some aspects of Fileset which are not completely sound or correct.
 * For instance, in FilesetEntries you may specify package sources as:
 * "//path/to/package". Everywhere else, this refers to the default rule
 * in package "//path/to/package". However, in a FilesetEntry, this refers
 * to the entire package.
 *
 * <p>Another issue is that links may cross package
 * boundaries if you link an entire tree recursively.
 * This is the behavior if the FilesetEntry has no "files" attribute.
 *
 * <p>The "src" attribute of a FilesetEntry may refer to an entire package
 * (as mentioned in the previous paragraph), the label of another
 * FilesetEntry rule.  It may also refer to a directory, even though this
 * is not sound.
 */
public class Fileset implements RuleConfiguredTargetFactory {

  /**
   * The collection of artifacts which must be built before the symlink
   * traversal may take place.
   */
  private static final String SYMLINK_TRAVERSAL_GUID = "f9df7978-457e-4ef5-85aa-5d281aa44495";
  private static final String FILE_TRAVERSAL_GUID = "798296d9-70b6-4f34-afa5-54eae79b989e";

  @Override
  public RuleConfiguredTarget create(RuleContext ruleContext) {
    PathFragment inputManifestPath = Util.getWorkspaceRelativePath(ruleContext.getTarget(), "",
        ".fileset_manifest");
    Artifact inputManifest = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        inputManifestPath, ruleContext.getConfiguration().getBinDirectory());
    NestedSetBuilder<Artifact> deps = NestedSetBuilder.compileOrder();

    List<SymlinkTraversal> traversals = new ArrayList<>();
    for (ConfiguredFilesetEntry entry : ruleContext.getFilesetEntryMap().get("entries")) {
      String errorMsg = entry.getEntry().validate();
      if (errorMsg != null) {
        ruleContext.attributeError("entries", errorMsg);
        return null;
      }

      checkExcludes(ruleContext, entry.getEntry());

      SymlinkTraversal traversal = addLinksForEntry(ruleContext, entry, deps);
      if (traversal != null) {
        traversals.add(traversal);
      }
    }

    if (ruleContext.hasErrors()) {
      return null;
    }

    NestedSet<Artifact> filesToBuild = createSymlinkAction(ruleContext,
        inputManifest, traversals, ruleContext.attributes().get("out", Type.OUTPUT), deps.build());

    if (filesToBuild == null) {
      return null;
    }

    Runfiles runfiles = new Runfiles.Builder()
        .addRunfiles(RunfilesCollector.State.DEFAULT, ruleContext)
        .addArtifacts(filesToBuild)
        .build();

    return new GenericRuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .add(RunfilesProvider.class, new StatelessRunfilesProvider(runfiles))
        .add(FilesetProvider.class,
            new FilesetProviderImpl(inputManifest, getFilesetLinkDir(ruleContext)))
        .build();
  }

  private void checkExcludes(RuleContext ruleContext, FilesetEntry entry) {
    List<String> excludes = entry.getExcludes();
    if (excludes == null) {
      return;
    }

    for (String exclude : excludes) {
      PathFragment excludeFrag = new PathFragment(exclude);
      if (excludeFrag.segmentCount() > 1) {
        ruleContext.attributeWarning("entries", "Exclude " + exclude + " will be ignored. "
            + "Excludes only apply one level deep");
      }
    }
  }

  /**
   * Creates the traversal for the given FilesetEntry.
   * @return the traversal, or null if there was an error.
   */
  @Nullable
  private SymlinkTraversal addLinksForEntry(
      RuleContext ruleContext, ConfiguredFilesetEntry entry, NestedSetBuilder<Artifact> deps) {
    return (entry.getFiles() == null)
        ? recursiveTraversal(ruleContext, entry, deps)
        : fileListTraversal(ruleContext, entry, deps);
  }

  /**
   * Create the traversal in the case where the entry has specified
   * a list of files.
   */
  private SymlinkTraversal fileListTraversal(RuleContext ruleContext,
      final ConfiguredFilesetEntry configuredEntry, final NestedSetBuilder<Artifact> deps) {
    final FilesetEntry entry = configuredEntry.getEntry();
    final Map<Artifact, PathFragment> files = new LinkedHashMap<>();
    final PathFragment stripPrefix = entry.getStripPrefix().equals(".")
        ? null
        : entry.getSrcLabel().getPackageFragment().getRelative(entry.getStripPrefix());
    final boolean recursive =
        ruleContext.getConfiguration().getCheckFilesetDependenciesRecursively();
    boolean containsErrors = false;

    for (TransitiveInfoCollection target : configuredEntry.getFiles()) {
      // Sort filesToBuild for consistency across undefined iteration order.
      Collection<Artifact> filesToBuild = new Ordering<Artifact>() {
        @Override
        public int compare(Artifact a1, Artifact a2) {
          return a1.getPath().compareTo(a2.getPath());
        }
      }.immutableSortedCopy(target.getProvider(FileProvider.class).getFilesToBuild());
      if (filesToBuild.size() == 1 && stripPrefix == null) {
        files.put(Iterables.getOnlyElement(filesToBuild),
                  new PathFragment(target.getLabel().getName()));
      } else {
        for (Artifact artifact : filesToBuild) {
          PathFragment execPath = artifact.getRootRelativePath();
          if (stripPrefix != null) {
            if (!execPath.startsWith(stripPrefix)) {
              ruleContext.attributeError("entries", String.format(
                  "Artifact '%s' is not under the specified strip prefix '%s'",
                  execPath, stripPrefix));
              containsErrors = true;
            } else {
              files.put(artifact, execPath.relativeTo(stripPrefix));
            }
          } else {
            files.put(artifact, new PathFragment(execPath.getBaseName()));
          }
        }
      }

      // Preserve addition order: add as transitive set rather than direct elements.
      deps.addTransitive(NestedSetBuilder.wrap(Order.COMPILE_ORDER, filesToBuild));
    }

    if (containsErrors) {
      return null;
    }

    final Label label = ruleContext.getLabel();
    final ActionOwner owner = ruleContext.getActionOwner();
    return new SymlinkTraversal() {
      @Override
      public void addSymlinks(ErrorEventListener listener, FilesetLinks links,
          ThreadPoolExecutor filesetPool) throws IOException, InterruptedException {
        for (Map.Entry<Artifact, PathFragment> fileEntry : files.entrySet()) {
          Artifact artifact = fileEntry.getKey();
          Path target = artifact.getPath();
          PathFragment src = entry.getDestDir().getRelative(fileEntry.getValue());

          if (!artifact.isSourceArtifact() && target.isDirectory(Symlinks.FOLLOW)) {
            listener.warn(null,
                String.format("Fileset '%s' file path '%s' is a directory, which is unsound",
                              label, artifact.prettyPrint()));
          }

          FilesetUtil.collectFilesNoExcludes(filesetPool, target, src, links, listener,
              owner, new FilesetUtil.VisitParameters(recursive, entry.getSymlinkBehavior()));
        }
      }

      @Override
      public void fingerprint(Fingerprint fp) {
        fp.addString(SYMLINK_TRAVERSAL_GUID);
        if (stripPrefix == null) {
          // Note: "#" should not appear in non-empty prefixes.
          fp.addString("#NO_STRIP#");
        } else {
          fp.addPath(stripPrefix);
        }
        for (Map.Entry<Artifact, PathFragment> fileEntry : files.entrySet()) {
          fp.addPath(fileEntry.getKey().getPath());
          fp.addPath(fileEntry.getValue());

        }
        fp.addInt(recursive ? 1 : 0);
        fingerprintFilesetEntry(fp, entry);
      }

      @Override
      public boolean executeUnconditionally(PackageUpToDateChecker upToDateChecker) {
        // Note: isVolatile must return true if executeUnconditionally can ever return true
        // for this instance.
        for (TransitiveInfoCollection target : configuredEntry.getFiles()) {
          for (Artifact artifact : target.getProvider(FileProvider.class).getFilesToBuild()) {
            if (!isUnmodifiedFile(artifact, upToDateChecker) && !artifact.getPath().isFile()) {
              // If the path is a regular file, it triggers the normal modes
              // of dependency checking. Otherwise, we're doing an implicit
              // recursive traversal: Dependency checking of directories is
              // unsound, so we work around that via unconditionality.
              return true;
            }
          }
        }
        return false;
      }

      @Override
      public boolean isVolatile() {
        return true;
      }
    };
  }

  private static boolean isUnmodifiedFile(Artifact artifact,
      PackageUpToDateChecker upToDateChecker) {
    return artifact.getOwner() != null &&
        !upToDateChecker.loadedTargetMayHaveChanged(artifact.getOwner());
  }

  private static <T> Collection<T> nonNullCollection(Collection<T> collection) {
    return (collection == null) ? Collections.<T>emptySet() : collection;
  }

  private SymlinkTraversal createRecursiveTraversal(final RuleContext ruleContext,
      final FilesetEntry entry, final Path srcDir, TransitiveInfoCollection target,
      NestedSetBuilder<Artifact> deps) {
    final NestedSet<Artifact> files = target.getProvider(FileProvider.class).getFilesToBuild();
    deps.addTransitive(files);

    final boolean recursive =
        ruleContext.getConfiguration().getCheckFilesetDependenciesRecursively();

    // This will compute a recursive file traversal, and will be called at
    // execution time.
    return new SymlinkTraversal() {
      @Override
      public void addSymlinks(ErrorEventListener listener, FilesetLinks filesetLinks,
          ThreadPoolExecutor filesetPool) throws IOException, InterruptedException {
        Collection<String> excludes = nonNullCollection(entry.getExcludes());
        SubpackageMode pkgMode = ruleContext.getConfiguration().isStrictFilesets()
            ? SubpackageMode.ERROR
            : SubpackageMode.WARNING;
        FilesetUtil.collectFilesRecursively(
            filesetPool,
            srcDir,
            entry.getDestDir(), filesetLinks, excludes,
            /*pkgMode=*/entry.isSourceFileset() ? pkgMode : SubpackageMode.IGNORE,
            listener, ruleContext.getActionOwner(),
            new FilesetUtil.VisitParameters(recursive, entry.getSymlinkBehavior()));
      }

      @Override
      public void fingerprint(Fingerprint fp) {
        fp.addString(FILE_TRAVERSAL_GUID);
        fp.addPath(srcDir);
        fp.addInt(recursive ? 1 : 0);
        fp.addInt(ruleContext.getConfiguration().isStrictFilesets() ? 1 : 0);
        fingerprintFilesetEntry(fp, entry);
      }

      @Override
      public boolean executeUnconditionally(PackageUpToDateChecker upToDateChecker) {
        // Note: isVolatile must return true if executeUnconditionally can ever return true
        // for this instance.
        for (Artifact artifact : files) {
          if (!isUnmodifiedFile(artifact, upToDateChecker)) {
            // We're doing a recursive traversal: We need to
            // disable dependency checking on this action because a Fileset may
            //  link to all files in a package (a recursive traversal starting
            // at the package directory).
            return true;
          }
        }
        return false;
      }

      @Override
      public boolean isVolatile() {
        return true;
      }

    };
  }

  private static void fingerprintFilesetEntry(Fingerprint fp, FilesetEntry entry) {
    // Note taht differences in the file list or srcdir are put into the
    // fingerprint by the caller.

    fp.addPath(entry.getDestDir());
    if (entry.getExcludes() == null) {
      // Note: "#" should not appear in non-empty excludes.
      fp.addString("#NO_EXCLUDES#");
    } else {
      fp.addStrings(entry.getExcludes());
    }
    fp.addString(entry.getSymlinkBehavior().name());
    fp.addInt(entry.isSourceFileset() ? 1 : 0);
  }

  private SymlinkTraversal createFilesetManifestTraversal(RuleContext ruleContext,
      TransitiveInfoCollection filesetTarget, FilesetEntry entry, NestedSetBuilder<Artifact> deps) {
    FilesetProvider filesetProvider = filesetTarget.getProvider(FilesetProvider.class);
    Preconditions.checkArgument(filesetProvider != null);
    // TODO(bazel-team): This avoids the FilesetManifestTraversal optimization in
    // the --nocheck_fileset_dependencies_recursively case. This might not
    // be so bad because in this case, the output tree should be smaller anyway.
    // We could achieve both optimizations simultaneously by creating a
    // Filesystem adapter for deserialized Fileset manifests which could be
    // plugged into the traversal logic in FilesetUtil.
    if (!ruleContext.getConfiguration().getCheckFilesetDependenciesRecursively()) {
      PathFragment linkDir = filesetProvider.getFilesetLinkDir();
      Path srcPath =
          ruleContext.getConfiguration().getBinDirectory().getPath().getRelative(linkDir);
      return createRecursiveTraversal(ruleContext, entry, srcPath, filesetTarget, deps);
    }

    Artifact inputManifest = filesetProvider.getFilesetInputManifest();
    deps.addTransitive(filesetTarget.getProvider(FileProvider.class).getFilesToBuild());
    return new FilesetManifestTraversal(inputManifest.getPath(), entry.getDestDir(),
                                        nonNullCollection(entry.getExcludes()));
  }

  /**
   * Create the traversal in the case where must recursively descend a
   * directory. We must do the traversal at execution-time.
   */
  private SymlinkTraversal recursiveTraversal(RuleContext ruleContext,
      final ConfiguredFilesetEntry configuredEntry, NestedSetBuilder<Artifact> deps) {
    FilesetEntry entry = configuredEntry.getEntry();
    TransitiveInfoCollection srcTarget = configuredEntry.getSrc();

    if (srcTarget.getProvider(FilesetProvider.class) != null) {
      return createFilesetManifestTraversal(ruleContext, srcTarget, entry, deps);
    } else if (entry.isSourceFileset()) {
      // We figure out the package directory from the location of the BUILD file. If this condition
      // is true, the artifact must be one.
      Path buildFilePath = Iterables.getOnlyElement(
          srcTarget.getProvider(FileProvider.class).getFilesToBuild()).getPath();
      Preconditions.checkState(buildFilePath.getBaseName().equals("BUILD"));
      return createRecursiveTraversal(
          ruleContext, entry, buildFilePath.getParentDirectory(), srcTarget, deps);
    } else {
      // At this point the srcdir must be an input or an output file. This is assured by
      // RuleContext.validateFilesetEntry() .
      Artifact artifact = Iterables.getOnlyElement(
          srcTarget.getProvider(FileProvider.class).getFilesToBuild());
      return createRecursiveTraversal(ruleContext, entry, artifact.getPath(), srcTarget, deps);
    }
  }

  private PathFragment getFilesetLinkDir(RuleContext ruleContext) {
    Label out = ruleContext.attributes().get("out", Type.OUTPUT);
    return Util.getWorkspaceRelativePath(ruleContext.getTarget()).replaceName(out.getName());
  }


  private NestedSet<Artifact> createSymlinkAction(RuleContext ruleContext,
      Artifact inputManifest, List<SymlinkTraversal> traversals, final Label out,
      NestedSet<Artifact> deps) {
    // This is similar to creating the runfiles tree.
    // In fact, we reuse the embedded script "build-runfiles.cc" becuase
    // the use case is so similar.  The "build-runfiles" script
    // creates a directory structure where the "MANIFEST" file and
    // the workspace directory are siblings. We accomodate this by creating a symlink
    // from the intended output path to the workspace root.

    BuildConfiguration config = ruleContext.getConfiguration();
    Root bindir = config.getBinDirectory();

    PathFragment pkgDir = ruleContext.getLabel().getPackageFragment();
    PathFragment outFrag = new PathFragment(out.getName());
    if (outFrag.startsWith(new PathFragment("."))) {
      ruleContext.attributeError("out", "Invalid output label '" + out + "'. " +
                     "Output path must reside below the package. Do not use '.'");
      return null;
    }

    PathFragment linkDir = pkgDir.getRelative(outFrag);
    PathFragment outputManifestFrag = AnalysisUtils.getManifestPathFromFilesetPath(linkDir);
    Artifact outputManifest = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
        outputManifestFrag, bindir);

    ruleContext.getAnalysisEnvironment().registerAction(new FilesetManifestAction(ruleContext
        .getActionOwner(), new CompoundSymlinkTraversal(traversals), deps, inputManifest));

    ruleContext.getAnalysisEnvironment().registerAction(new SymlinkTreeAction(
        ruleContext.getActionOwner(), inputManifest, outputManifest, config, /*filesetTree=*/true));

    final Artifact linkArtifact =
        ruleContext.getAnalysisEnvironment().getFilesetArtifact(linkDir, bindir);
    // Make sure we did not get another artifact created by someone else
    Preconditions.checkState(linkArtifact.isFileset());

    PathFragment workspaceRoot = outputManifest.getExecPath().getParentDirectory().getRelative(
        "google3");
    ruleContext.getAnalysisEnvironment().registerAction(new FilesetSymlinkAction(
        ruleContext.getActionOwner(), workspaceRoot, outputManifest, linkArtifact,
        "Symlinking " + ruleContext.getLabel()));
    return NestedSetBuilder.create(Order.STABLE_ORDER, linkArtifact);
  }

  private static final class FilesetSymlinkAction extends SymlinkAction {
    private FilesetSymlinkAction(ActionOwner owner, PathFragment inputPath, Artifact input,
      Artifact output, String progressMessage) {
      super(owner, inputPath, input, output, progressMessage);
    }

    @Override
    public void execute(ActionExecutionContext actionExecutionContext)
        throws ActionExecutionException {
      try {
        super.execute(actionExecutionContext);

        // Update mtime to force downstream re-execution of actions,
        // because dependency checking of Fileset output trees is
        // unsound (it's a directory).
        Path linkPath = getOutputPath();
        if (linkPath.exists()) {
          // -1L means "use the current time".
          linkPath.setLastModifiedTime(-1L);
        } else {
          // Should only happen if the Fileset included no links.
          actionExecutionContext.getExecutor().getExecRoot().getRelative(
              getInputPath()).createDirectory();
        }
      } catch (IOException e) {
        throw new ActionExecutionException("failed to touch symbolic link '"
            + Iterables.getOnlyElement(getOutputs()).prettyPrint()
            + "' to the '" + Iterables.getOnlyElement(getInputs()).prettyPrint()
            + "' due to I/O error: " + e.getMessage(), e, this, false);
      }
    }
  }
}
