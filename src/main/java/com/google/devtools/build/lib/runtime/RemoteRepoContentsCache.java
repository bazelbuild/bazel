package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** A remote cache for the contents of external repositories. */
public interface RemoteRepoContentsCache {
  /** Adds a repository that has been fetched locally to the remote cache. */
  void addToCache(
      RepositoryName repoName,
      Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws InterruptedException;

  /**
   * Retrieves a repository from the remote cache if possible.
   *
   * @return true if there was a cache hit and the repository has been fetched into the given
   *     directory.
   */
  boolean lookupCache(
      RepositoryName repoName,
      Path repoDir,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws IOException, InterruptedException;
}
