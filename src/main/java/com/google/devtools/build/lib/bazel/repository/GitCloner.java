// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.Status;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.api.errors.InvalidRefNameException;
import org.eclipse.jgit.api.errors.InvalidRemoteException;
import org.eclipse.jgit.api.errors.JGitInternalException;
import org.eclipse.jgit.api.errors.RefNotFoundException;
import org.eclipse.jgit.lib.Constants;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;
import org.eclipse.jgit.transport.NetRCCredentialsProvider;

import java.io.IOException;
import java.util.Objects;
import java.util.Set;

/**
 * Clones a Git repository, checks out the provided branch, tag, or commit, and
 * clones submodules if specified.
 */
public class GitCloner {
  private GitCloner() {
    // Only static methods in this class
  }

  private static boolean isUpToDate(GitRepositoryDescriptor descriptor) {
    // Initializing/checking status of/etc submodules cleanly is hard, so don't try for now.
    if (descriptor.initSubmodules) {
      return false;
    }
    Repository repository = null;
    try {
      repository = new FileRepositoryBuilder()
          .setGitDir(descriptor.directory.getChild(Constants.DOT_GIT).getPathFile())
          .setMustExist(true)
          .build();
      ObjectId head = repository.resolve(Constants.HEAD);
      ObjectId checkout = repository.resolve(descriptor.checkout);
      if (head != null && checkout != null && head.equals(checkout)) {
        Status status = Git.wrap(repository).status().call();
        if (!status.hasUncommittedChanges()) {
          // new_git_repository puts (only) BUILD and WORKSPACE, and
          // git_repository doesn't add any files.
          Set<String> untracked = status.getUntracked();
          if (untracked.isEmpty()
              || (untracked.size() == 2
                  && untracked.contains("BUILD")
                  && untracked.contains("WORKSPACE"))) {
            return true;
          }
        }
      }
    } catch (GitAPIException | IOException e) {
      // Any exceptions here, we'll just blow it away and try cloning fresh.
      // The fresh clone avoids any weirdness due to what's there and has nicer
      // error reporting.
    } finally {
      if (repository != null) {
        repository.close();
      }
    }
    return false;
  }

  public static SkyValue clone(Rule rule, Path outputDirectory, EventHandler eventHandler)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    if ((mapper.has("commit", Type.STRING) == mapper.has("tag", Type.STRING))
        && (mapper.get("commit", Type.STRING).isEmpty()
            == mapper.get("tag", Type.STRING).isEmpty())) {
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(), "One of either commit or tag must be defined"),
          Transience.PERSISTENT);
    }

    String startingPoint;
    if (mapper.has("commit", Type.STRING) && !mapper.get("commit", Type.STRING).isEmpty()) {
      startingPoint = mapper.get("commit", Type.STRING);
    } else {
      startingPoint = "tags/" + mapper.get("tag", Type.STRING);
    }

    GitRepositoryDescriptor descriptor = new GitRepositoryDescriptor(
        mapper.get("remote", Type.STRING),
        startingPoint,
        mapper.get("init_submodules", Type.BOOLEAN),
        outputDirectory);

    // Setup proxy if remote is http or https
    if (descriptor.remote != null && descriptor.remote.startsWith("http")) {
      try {
        ProxyHelper.createProxyIfNeeded(descriptor.remote);
      } catch (IOException ie) {
        throw new RepositoryFunctionException(ie, Transience.TRANSIENT);
      }
    }

    Git git = null;
    try {
      if (descriptor.directory.exists()) {
        if (isUpToDate(descriptor)) {
          return new HttpDownloadValue(descriptor.directory);
        }
        try {
          FileSystemUtils.deleteTree(descriptor.directory);
        } catch (IOException e) {
          throw new RepositoryFunctionException(e, Transience.TRANSIENT);
        }
      }
      git = Git.cloneRepository()
          .setURI(descriptor.remote)
          .setCredentialsProvider(new NetRCCredentialsProvider())
          .setDirectory(descriptor.directory.getPathFile())
          .setCloneSubmodules(false)
          .setNoCheckout(true)
          .setProgressMonitor(new GitProgressMonitor("Cloning " + descriptor.remote, eventHandler))
          .call();
      git.checkout()
          .setCreateBranch(true)
          .setName("bazel-checkout")
          .setStartPoint(descriptor.checkout)
          .call();

      // Using CloneCommand.setCloneSubmodules() results in SubmoduleInitCommand and
      // SubmoduleUpdateCommand to be called recursively for all submodules. This is not
      // desirable for repositories, such as github.com/rust-lang/rust-installer, which
      // recursively includes itself as a submodule, which would result in an infinite
      // loop if submodules are cloned recursively. For now, limit submodules to only
      // the first level.
      if (descriptor.initSubmodules && !git.submoduleInit().call().isEmpty()) {
        git.submoduleUpdate()
            .setProgressMonitor(
                new GitProgressMonitor(
                    "Cloning submodules for " + descriptor.remote, eventHandler))
            .call();
      }
    } catch (InvalidRemoteException e) {
      throw new RepositoryFunctionException(
          new IOException("Invalid Git repository URI: " + e.getMessage()), Transience.PERSISTENT);
    } catch (RefNotFoundException | InvalidRefNameException e) {
      throw new RepositoryFunctionException(
          new IOException("Invalid branch, tag, or commit: " + e.getMessage()),
          Transience.PERSISTENT);
    } catch (GitAPIException e) {
      // This is a sad attempt to actually get a useful error message out of jGit, which will bury
      // the actual (useful) cause of the exception under several throws.
      StringBuilder errmsg = new StringBuilder();
      errmsg.append(e.getMessage());
      Throwable throwable = e;
      while (throwable.getCause() != null) {
        throwable = throwable.getCause();
        errmsg.append(" caused by " + e.getMessage());
      }
      throw new RepositoryFunctionException(
          new IOException("Error cloning repository: " + errmsg), Transience.PERSISTENT);
    } catch (JGitInternalException e) {
      // This is a lame catch-all for jgit throwing RuntimeExceptions all over the place because,
      // as the docs put it, "a lot of exceptions are so low-level that is is unlikely that the
      // caller of the command can handle them effectively." Thanks, jgit.
      throw new RepositoryFunctionException(new IOException(e.getMessage()),
          Transience.PERSISTENT);
    } finally {
      if (git != null) {
        git.close();
      }
    }
    return new HttpDownloadValue(descriptor.directory);
  }

  private static final class GitRepositoryDescriptor {
    private final String remote;
    private final String checkout;
    private final boolean initSubmodules;
    private final Path directory;

    public GitRepositoryDescriptor(String remote, String checkout, boolean initSubmodules,
        Path directory) {
      this.remote = remote;
      this.checkout = checkout;
      this.initSubmodules = initSubmodules;
      this.directory = directory;
    }

    @Override
    public String toString() {
      return remote + " -> " + directory + " (" + checkout + ") submodules: "
          + initSubmodules;
    }

    @Override
    public boolean equals(Object obj) {
      if (obj == this) {
        return true;
      }
      if (!(obj instanceof GitRepositoryDescriptor)) {
        return false;
      }
      GitRepositoryDescriptor other = (GitRepositoryDescriptor) obj;
      return Objects.equals(remote, other.remote)
          && Objects.equals(checkout, other.checkout)
          && Objects.equals(initSubmodules, other.initSubmodules)
          && Objects.equals(directory, other.directory);
    }

    @Override
    public int hashCode() {
      return Objects.hash(remote, checkout, initSubmodules, directory);
    }
  }
}
