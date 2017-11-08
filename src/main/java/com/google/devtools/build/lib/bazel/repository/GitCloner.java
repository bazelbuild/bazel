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

import com.google.common.base.Ascii;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.net.UrlEscapers;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.bazel.repository.downloader.ProxyHelper;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.FetchEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.net.URL;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
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

/**
 * Clones a Git repository, checks out the provided branch, tag, or commit, and
 * clones submodules if specified.
 */
public class GitCloner {

  private static final Pattern GITHUB_URL = Pattern.compile(
      "(?:git@|https?://)github\\.com[:/](\\w+)/(\\w+)\\.git");

  private static final Pattern GITHUB_VERSION_FORMAT = Pattern.compile("v(\\d+\\.)*\\d+");

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

  public static HttpDownloadValue clone(
      Rule rule,
      Path outputDirectory,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnvironment,
      HttpDownloader downloader)
      throws RepositoryFunctionException {
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    if (mapper.isAttributeValueExplicitlySpecified("commit")
        == mapper.isAttributeValueExplicitlySpecified("tag")) {
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(), "One of either commit or tag must be defined"),
          Transience.PERSISTENT);
    }

    GitRepositoryDescriptor descriptor;
    try {
      if (mapper.isAttributeValueExplicitlySpecified("commit")) {
        descriptor = GitRepositoryDescriptor.createWithCommit(
            mapper.get("remote", Type.STRING),
            mapper.get("commit", Type.STRING),
            mapper.get("init_submodules", Type.BOOLEAN),
            outputDirectory);
      } else {
        descriptor = GitRepositoryDescriptor.createWithTag(
            mapper.get("remote", Type.STRING),
            mapper.get("tag", Type.STRING),
            mapper.get("init_submodules", Type.BOOLEAN),
            outputDirectory);
      }
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }

    // Setup proxy if remote is http or https
    if (descriptor.remote != null && Ascii.toLowerCase(descriptor.remote).startsWith("http")) {
      try {
        new ProxyHelper(clientEnvironment).createProxyIfNeeded(new URL(descriptor.remote));
      } catch (IOException ie) {
        throw new RepositoryFunctionException(ie, Transience.TRANSIENT);
      }
    }

    BuildEvent fetchEvent = null;
    Git git = null;
    Exception suppressedException = null;
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

      String uncheckedSha256 = getUncheckedSha256(mapper);
      if (repositoryLooksTgzable(descriptor.remote)) {
        Optional<Exception> maybeException = downloadRepositoryAsHttpArchive(
            descriptor, eventHandler, clientEnvironment, downloader, uncheckedSha256);
        if (maybeException.isPresent()) {
          suppressedException = maybeException.get();
        } else {
          return new HttpDownloadValue(descriptor.directory);
        }
      }
      if (!Strings.isNullOrEmpty(uncheckedSha256)) {
        // Specifying a sha256 forces this to use a tarball download.
        IOException e = new IOException(
            "Could not download tarball, but sha256 specified (" + uncheckedSha256 + ")");
        if (suppressedException != null) {
          e.addSuppressed(suppressedException);
        }
        throw new RepositoryFunctionException(e, Transience.TRANSIENT);
      }

      fetchEvent = new FetchEvent(descriptor.remote.toString(), false);
      git =
          Git.cloneRepository()
              .setURI(descriptor.remote)
              .setCredentialsProvider(new NetRCCredentialsProvider())
              .setDirectory(descriptor.directory.getPathFile())
              .setCloneSubmodules(false)
              .setNoCheckout(true)
              .setProgressMonitor(
                  new GitProgressMonitor(
                      descriptor.remote, "Cloning " + descriptor.remote, eventHandler))
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
                    descriptor.remote, "Cloning submodules for " + descriptor.remote, eventHandler))
            .call();
      }
      fetchEvent = new FetchEvent(descriptor.remote.toString(), true);
    } catch (InvalidRemoteException e) {
      if (suppressedException != null) {
        e.addSuppressed(suppressedException);
      }
      throw new RepositoryFunctionException(
          new IOException("Invalid Git repository URI: " + e.getMessage()), Transience.PERSISTENT);
    } catch (RefNotFoundException | InvalidRefNameException e) {
      if (suppressedException != null) {
        e.addSuppressed(suppressedException);
      }
      throw new RepositoryFunctionException(
          new IOException("Invalid branch, tag, or commit: " + e.getMessage()),
          Transience.PERSISTENT);
    } catch (GitAPIException e) {
      if (suppressedException != null) {
        e.addSuppressed(suppressedException);
      }
      // This is a sad attempt to actually get a useful error message out of jGit, which will bury
      // the actual (useful) cause of the exception under several throws.
      StringBuilder errmsg = new StringBuilder();
      errmsg.append(e.getMessage());
      Throwable throwable = e;
      while (throwable.getCause() != null) {
        throwable = throwable.getCause();
        errmsg.append(" caused by " + throwable.getMessage());
      }
      throw new RepositoryFunctionException(
          new IOException("Error cloning repository: " + errmsg), Transience.PERSISTENT);
    } catch (JGitInternalException e) {
      if (suppressedException != null) {
        e.addSuppressed(suppressedException);
      }
      // This is a lame catch-all for jgit throwing RuntimeExceptions all over the place because,
      // as the docs put it, "a lot of exceptions are so low-level that is is unlikely that the
      // caller of the command can handle them effectively." Thanks, jgit.
      throw new RepositoryFunctionException(new IOException(e.getMessage()),
          Transience.PERSISTENT);
    } finally {
      if (git != null) {
        git.close();
      }
      if (fetchEvent != null) {
        eventHandler.post(fetchEvent);
      }
    }
    return new HttpDownloadValue(descriptor.directory);
  }

  private static String getUncheckedSha256(WorkspaceAttributeMapper mapper)
      throws RepositoryFunctionException {
    if (mapper.isAttributeValueExplicitlySpecified("sha256")) {
      try {
        return mapper.get("sha256", Type.STRING);
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    }
    return "";
  }

  private static boolean repositoryLooksTgzable(String remote) {
    // Only handles GitHub right now.
    return GITHUB_URL.matcher(remote).matches();
  }

  private static Optional<Exception> downloadRepositoryAsHttpArchive(
      GitRepositoryDescriptor descriptor, ExtendedEventHandler eventHandler,
      Map<String, String> clientEnvironment, HttpDownloader downloader, String uncheckedSha256)
      throws RepositoryFunctionException {
    Matcher matcher = GITHUB_URL.matcher(descriptor.remote);
    Preconditions.checkState(
        matcher.matches(), "Remote should be checked before calling this method");
    String user = matcher.group(1);
    String repositoryName = matcher.group(2);
    String downloadUrl =
        "https://github.com/"
            + UrlEscapers.urlFragmentEscaper()
                .escape(user + "/" + repositoryName + "/archive/" + descriptor.ref + ".tar.gz");
    try {
      FileSystemUtils.createDirectoryAndParents(descriptor.directory);
      Path tgz = downloader.download(ImmutableList.of(new URL(downloadUrl)), uncheckedSha256,
          Optional.of("tar.gz"), descriptor.directory, eventHandler, clientEnvironment);
      String githubRef = descriptor.ref;
      if (githubRef.startsWith("v") && GITHUB_VERSION_FORMAT.matcher(githubRef).matches()) {
        githubRef = githubRef.substring(1);
      }
      DecompressorValue.decompress(
          DecompressorDescriptor.builder()
              .setArchivePath(tgz)
              // GitHub puts the contents under a directory called <repo>-<commit>.
              .setPrefix(repositoryName + "-" + githubRef)
              .setRepositoryPath(descriptor.directory)
              .build());
    } catch (InterruptedException | IOException e) {
      try {
        FileSystemUtils.deleteTree(descriptor.directory);
      } catch (IOException e1) {
        throw new RepositoryFunctionException(
            new IOException("Unable to delete " + descriptor.directory + ": " + e1.getMessage()),
            Transience.TRANSIENT);
      }
      return Optional.<Exception>of(e);
    }
    return Optional.absent();
  }

  private static final class GitRepositoryDescriptor {
    private final String remote;
    private final String checkout;
    private final boolean initSubmodules;
    private final Path directory;
    private final String ref;

    private GitRepositoryDescriptor(String remote, String ref, String checkout,
        boolean initSubmodules, Path directory) {
      this.remote = remote;
      this.ref = ref;
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
          && Objects.equals(ref, other.ref)
          && Objects.equals(initSubmodules, other.initSubmodules)
          && Objects.equals(directory, other.directory);
    }

    @Override
    public int hashCode() {
      return Objects.hash(remote, ref, initSubmodules, directory);
    }

    static GitRepositoryDescriptor createWithCommit(String remote, String commit,
        Boolean initSubmodules, Path outputDirectory) {
      return new GitRepositoryDescriptor(
          remote, commit, commit, initSubmodules, outputDirectory);
    }

    static GitRepositoryDescriptor createWithTag(String remote, String tag,
        Boolean initSubmodules, Path outputDirectory) {
      return new GitRepositoryDescriptor(
          remote, tag, "tags/" + tag, initSubmodules, outputDirectory);
    }
  }
}
