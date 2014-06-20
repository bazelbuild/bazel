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

import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.syntax.FilesetEntry;
import com.google.devtools.build.lib.vfs.IORuntimeException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;

import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Static utility methods for Filesets.
 *
 * <p>These methods are implementation details of Fileset, so the class is package-private.
 */
class FilesetUtil {
  public static class VisitParameters {
    private final boolean checkDepsRecursively;
    private final FilesetEntry.SymlinkBehavior symlinkBehavior;

    public VisitParameters(boolean checkDepsRecursively,
                           FilesetEntry.SymlinkBehavior symlinkBehavior) {
      this.checkDepsRecursively = checkDepsRecursively;
      this.symlinkBehavior = symlinkBehavior;
    }

    public boolean getCheckDepsRecursively() {
      return checkDepsRecursively;
    }

    public FilesetEntry.SymlinkBehavior getSymlinkBehavior() {
      return symlinkBehavior;
    }
  }

  /** How to handle filesets that cross subpackages. */
  public static enum SubpackageMode {
    ERROR, WARNING, IGNORE;
  }

  private FilesetUtil() {
  }

  /**
   * Get all the files recursively under the given Path, following symlinks.
   *
   * @param pool an execution-phase shared threadpool for fileset visitation.
   * @param root the root directory.
   * @param src the symlink source.
   * @param links the fileset links to add to.
   * @param excludes paths to exclude.
   * @param pkgMode how to handle recursing into a BUILD file directory.
   * @param owner the ActionOwner.
   * @param visitParameters Visit parameters.
   * @throws IOException if a filesystem operation fails.
   * @throws InterruptedException if the thread is interrupted.
   */
  @ThreadSafety.ThreadSafe
  static void collectFilesRecursively(ThreadPoolExecutor pool, Path root, PathFragment src,
      FilesetLinks links, Collection<String> excludes, SubpackageMode pkgMode,
      ErrorEventListener listener, ActionOwner owner, VisitParameters visitParameters)
      throws IOException, InterruptedException {
    FilesetVisitor visitor = new FilesetVisitor(pool);
    visitor.visitLinksRecursively(root, src, links, excludes, pkgMode, listener, owner,
        visitParameters);
    visitor.work();
  }


  /**
   * Similar to collectFilesRecursively above, but only applies to non top-level directories.
   * Because we don't allow excludes here, we do not need to expand directory symlinks.
   *
   * @param target the symlink target.
   * @param src    the symlink source.
   * @param links  the fileset links to add to.
   * @param owner  the ActionOwner.
   * @param visitParameters Visit parameters.
   * @throws IOException if a filesystem operation fails.
   * @throws InterruptedException if the thread is interrupted.
   */
  @ThreadSafety.ThreadSafe
  static void collectFilesNoExcludes(ThreadPoolExecutor pool, Path target, PathFragment src,
      FilesetLinks links, ErrorEventListener listener, ActionOwner owner,
      VisitParameters visitParameters) throws IOException, InterruptedException {
    FilesetVisitor visitor = new FilesetVisitor(pool);
    // We use IGNORE here due to spurious warnings when Fileset "a" depends on Fileset "b", and
    // "b" contains a BUILD file in its output tree.
    // This may or may not be correct.
    visitor.visitLinkNoExcludes(target, src, links, SubpackageMode.IGNORE, listener, owner,
        visitParameters);
    visitor.work();
  }

  /**
   * A queue visitor which does a recursive traversal of a source directory,
   * adding symlinks as it goes.
   */
  @ThreadSafety.ThreadSafe
  private static final class FilesetVisitor extends AbstractQueueVisitor {
    public FilesetVisitor(ThreadPoolExecutor pool) {
      super(pool, /*shutdownOnCompletion=*/false, /*failFastOnException=*/true,
            /*failFastOnInterrupt=*/true);
    }

    private void enqueue(final Path target, final PathFragment src,
        final FilesetLinks links, final SubpackageMode pkgMode, final ErrorEventListener listener,
        final ActionOwner owner, final VisitParameters visitParameters) {
      super.enqueue(new Runnable() {
        @Override
        public void run() {
          try {
            visitLinkNoExcludes(target, src, links, pkgMode, listener, owner, visitParameters);
          } catch (IOException e) {
            throw new IORuntimeException(e);
          }
        }
      });
    }

    /**
     * Visit all directory entries for a given directory.
     *
     * @param target The directory to recurse into.
     * @param src    The location in the tree to which to write the symlinks.
     * @param links  The FilesetLinks object that tracks all links.
     * @param pkgMode How to handle recursing into another package.
     * @param owner  The ActionOwner.
     * @param visitParameters Visit parameters.
     * @throws IOException If a filesystem operation fails.
     */
    private void visitDirectoryEntries(Path target, PathFragment src, FilesetLinks links,
        SubpackageMode pkgMode, ErrorEventListener listener, ActionOwner owner,
        VisitParameters visitParameters) throws IOException {
      for (Path element : target.getDirectoryEntries()) {
        // TODO(bazel-team): Consider the case where the BUILD file exists on another
        // package path.
        if ((pkgMode != SubpackageMode.IGNORE) && element.isFile() &&
            element.getBaseName().equals("BUILD")) {
          // We must allow this for parity with the subinclude-based Fileset.
          String msg = "Fileset crosses package boundary into package rooted at " + target;
          if (pkgMode == SubpackageMode.WARNING) {
            listener.warn(owner.getLocation(), msg);
          } else {
            listener.error(owner.getLocation(), msg);
            throw new BadSubpackageException(msg);
          }

          // Don't need to warn on nested subpackages.
          pkgMode = SubpackageMode.IGNORE;
        }
        enqueue(element, src.getRelative(element.getBaseName()), links, pkgMode, listener, owner,
            visitParameters);
      }
    }

    /**
     * Visit all late parent directories (including src itself). This function
     * will find any late directories we haven't recursed into and add their
     * contents. Also, it will add a stub late directory for each parent dir
     * (but not src itself unless addStubLateDir is true) in case that parent
     * dir is encountered later.
     *
     * @param src    The location in the tree to which to write the symlinks.
     * @param links  The FilesetLinks object that tracks all links.
     * @param owner  The ActionOwner.
     * @param visitParameters Visit parameters.
     * @param addStubLateDir If there is no late dir for src, add a stub saying
     *     it has already been scanned.
     * @throws IOException If a filesystem operation fails.
     */
    private boolean visitLateParentDirs(PathFragment src, FilesetLinks links,
        ErrorEventListener listener, ActionOwner owner, VisitParameters visitParameters,
        boolean addStubLateDir) throws IOException {
      boolean addSucceeded = false;
      FilesetLinks.LateDirectoryInfo lateDir = links.getLateDirectoryInfo(src);
      if (lateDir == null && addStubLateDir) {
        lateDir = FilesetLinks.LateDirectoryInfo.createStub();
        addSucceeded = links.putLateDirectoryInfo(src, lateDir);
        if (!addSucceeded) {
          lateDir = links.getLateDirectoryInfo(src);
        }
      }
      if (lateDir != null) {
        if (lateDir.shouldAdd()) {
          visitDirectoryEntries(lateDir.getTarget(), lateDir.getSrc(), links,
              lateDir.getPkgMode(), listener, owner, visitParameters);
        } else if (!addSucceeded) {
          return true;
        }
      }

      PathFragment parent = src.getParentDirectory();
      if (parent != null) {
        visitLateParentDirs(parent, links, listener, owner, visitParameters, true);
      }
      return lateDir != null;
    }

    /**
     * Similar to visitLinksRecursively above, but only applies to non
     * top-level directories. Because we don't allow excludes here, we
     * do not need to expand directory symlinks.
     *
     * @param target the symlink target.
     * @param src    the symlink source.
     * @param links  the fileset links to add to.
     * @param pkgMode how to handle recursing into a BUILD file directory.
     * @param owner  the ActionOwner.
     * @param visitParameters Visit parameters.
     * @throws IOException if a filesystem operation fails.
     */
    public void visitLinkNoExcludes(Path target, PathFragment src, FilesetLinks links,
        SubpackageMode pkgMode, ErrorEventListener listener, ActionOwner owner,
        VisitParameters visitParameters) throws IOException {
      boolean shouldRecurse = visitParameters.getCheckDepsRecursively() ||
          visitLateParentDirs(src, links, listener, owner, visitParameters, false);
      Symlinks followSymlinks =
        visitParameters.getCheckDepsRecursively() ? Symlinks.NOFOLLOW : Symlinks.FOLLOW;

      if (target.isDirectory(followSymlinks)) {
        if (!shouldRecurse) {
          FilesetLinks.LateDirectoryInfo lateDir =
              FilesetLinks.LateDirectoryInfo.create(
                  target, src, pkgMode, getMetadata(target),
                  visitParameters.getSymlinkBehavior());
          // Could have been added when we weren't looking.
          if (!links.putLateDirectoryInfo(src, lateDir)) {
            shouldRecurse = true;
            visitLateParentDirs(src, links, listener, owner, visitParameters, false);
          }
        }
        if (shouldRecurse) {
          visitDirectoryEntries(target, src, links, pkgMode, listener, owner, visitParameters);
        }
      } else {
        links.addFile(src, target, getMetadata(target), visitParameters.getSymlinkBehavior());
      }
    }

    /**
     * Gets the metadata for the given target.
     */
    private String getMetadata(Path target) {
      try {
        return Long.toString(target.getLastModifiedTime());
      } catch (IOException e) {
        // Sometimes relative links don't point to actual files.
        // We must be able to proceed in this case.
        return "-1";
      }
    }

    /**
     * Visits all the files recursively under the given Path, following symlinks.
     * Symlinks are added to links as it goes.
     *
     * @param root the root directory.
     * @param src the symlink source.
     * @param links the fileset links to add to.
     * @param excludes paths to exclude.
     * @param pkgMode how to handle recursing into a BUILD file directory.
     * @param owner the ActionOwner.
     * @param visitParameters Visit parameters.
     * @throws IOException if a filesystem operation fails.
     */
    public void visitLinksRecursively(Path root, PathFragment src, FilesetLinks links,
        Collection<String> excludes, SubpackageMode pkgMode, ErrorEventListener listener,
        ActionOwner owner, VisitParameters visitParameters) throws IOException {
      for (Path element : root.getDirectoryEntries()) {
        if (!excludes.contains(element.getBaseName())) {
          // Excludes only apply to the top-level elements, so this is ok.
          enqueue(element, src.getRelative(element.getBaseName()), links, pkgMode, listener,
              owner, visitParameters);
        }
      }
    }

    public void work() throws InterruptedException, IOException {
      try {
        super.work(false);
      } catch (IORuntimeException e) {
        if (Thread.interrupted()) {
          // As per the contract of AbstractQueueVisitor#work, if an unchecked exception is thrown
          // and the build is interrupted, the thrown exception is what will be rethrown. Since the
          // user presumably wanted to interrupt the build, we ignore the thrown IORuntimeException
          // (which doesn't indicate a programming bug) and throw an InterruptedException.
          throw new InterruptedException();
        }
        throw e.getCauseIOException();
      }
    }
  }
}
