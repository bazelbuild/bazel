// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auth.Credentials;
import com.google.common.base.MoreObjects;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCacheHitEvent;
import com.google.devtools.build.lib.bazel.repository.downloader.UrlRewriter.RewrittenURL;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Bazel file downloader.
 *
 * <p>This class uses a {@link Downloader} to download files from external mirrors and writes them
 * to disk.
 */
public class DownloadManager {

  private final RepositoryCache repositoryCache;
  private List<Path> distdir = ImmutableList.of();
  private UrlRewriter rewriter;
  private final Downloader downloader;
  private boolean disableDownload = false;
  private int retries = 0;
  private boolean urlsAsDefaultCanonicalId;
  @Nullable private Credentials netrcCreds;

  public DownloadManager(RepositoryCache repositoryCache, Downloader downloader) {
    this.repositoryCache = repositoryCache;
    this.downloader = downloader;
  }

  public void setDistdir(List<Path> distdir) {
    this.distdir = ImmutableList.copyOf(distdir);
  }

  public void setUrlRewriter(UrlRewriter rewriter) {
    this.rewriter = rewriter;
  }

  public void setDisableDownload(boolean disableDownload) {
    this.disableDownload = disableDownload;
  }

  public void setRetries(int retries) {
    checkArgument(retries >= 0, "Invalid retries");
    this.retries = retries;
  }

  public void setUrlsAsDefaultCanonicalId(boolean urlsAsDefaultCanonicalId) {
    this.urlsAsDefaultCanonicalId = urlsAsDefaultCanonicalId;
  }

  public void setNetrcCreds(Credentials netrcCreds) {
    this.netrcCreds = netrcCreds;
  }

  /**
   * Downloads file to disk and returns path.
   *
   * <p>If the checksum and path to the repository cache is specified, attempt to load the file from
   * the {@link RepositoryCache}. If it doesn't exist, proceed to download the file and load it into
   * the cache prior to returning the value.
   *
   * @param originalUris list of mirror URIs with identical content
   * @param checksum valid checksum which is checked, or absent to disable
   * @param type extension, e.g. "tar.gz" to force on downloaded filename, or empty to not do this
   * @param output destination filename if {@code type} is <i>absent</i>, otherwise output directory
   * @param eventHandler CLI progress reporter
   * @param clientEnv environment variables in shell issuing this command
   * @param context the context in which the file was fetched; used only for reporting
   * @throws IllegalArgumentException on parameter badness, which should be checked beforehand
   * @throws IOException if download was attempted and ended up failing
   * @throws InterruptedException if this thread is being cast into oblivion
   */
  public Path download(
      List<URI> originalUris,
      Map<URI, Map<String, List<String>>> authHeaders,
      Optional<Checksum> checksum,
      String canonicalId,
      Optional<String> type,
      Path output,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      String context)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }

    if (Strings.isNullOrEmpty(canonicalId) && urlsAsDefaultCanonicalId) {
      Hasher hasher = DigestHashFunction.SHA256.getHashFunction().newHasher();
      for (URI uri : originalUris) {
        hasher.putString(uri.toASCIIString(), StandardCharsets.UTF_8);
      }
      canonicalId = hasher.hash().toString();
    }

    // TODO(andreisolo): This code path is inconsistent as the authHeaders are fetched from a
    //  .netrc only if it comes from a http_{archive,file,jar} - and it is handled directly
    //  by Starlark code -, or if a UrlRewriter is present. However, if it comes directly from a
    //  ctx.download{,_and_extract}, this not the case. Should be refactored to handle all .netrc
    //  parsing in one place, in Java code (similarly to #downloadAndReadOneUrl).
    ImmutableList<URI> rewrittenUrls = ImmutableList.copyOf(originalUris);
    Map<URI, Map<String, List<String>>> rewrittenAuthHeaders = authHeaders;

    if (rewriter != null) {
      ImmutableList<UrlRewriter.RewrittenURL> rewrittenUrlMappings = rewriter.amend(originalUris);
      rewrittenUrls =
          rewrittenUrlMappings.stream().map(url -> url.uri()).collect(toImmutableList());
      rewrittenAuthHeaders =
          rewriter.updateAuthHeaders(rewrittenUrlMappings, authHeaders, netrcCreds);
    }

    URI mainUri; // The "main" URI for this request
    // Used for reporting only and determining the file name only.
    if (rewrittenUrls.isEmpty()) {
      if (type.isPresent() && !Strings.isNullOrEmpty(type.get())) {
        mainUri = URI.create("http://nonexistent.example.org/cacheprobe." + type.get());
      } else {
        mainUri = URI.create("http://nonexistent.example.org/cacheprobe");
      }
    } else {
      mainUri = rewrittenUrls.get(0);
    }
    Path destination = getDownloadDestination(mainUri, type, output);
    ImmutableSet<String> candidateFileNames = getCandidateFileNames(mainUri, destination);

    // Is set to true if the value should be cached by the checksum value provided
    boolean isCachingByProvidedChecksum = false;

    if (checksum.isPresent()) {
      String cacheKey = checksum.get().toString();
      KeyType cacheKeyType = checksum.get().getKeyType();
      try {
        eventHandler.post(
            new CacheProgress(mainUri.toString(), "Checking in " + cacheKeyType + " cache"));
        String currentChecksum = RepositoryCache.getChecksum(cacheKeyType, destination);
        if (currentChecksum.equals(cacheKey)) {
          // No need to download.
          return destination;
        }
      } catch (IOException e) {
        // Ignore error trying to hash. We'll attempt to retrieve from cache or just download again.
      } finally {
        eventHandler.post(new CacheProgress(mainUri.toString()));
      }

      if (repositoryCache.isEnabled()) {
        isCachingByProvidedChecksum = true;

        try {
          Path cachedDestination =
              repositoryCache.get(cacheKey, destination, cacheKeyType, canonicalId);
          if (cachedDestination != null) {
            // Cache hit!
            eventHandler.post(new RepositoryCacheHitEvent(context, cacheKey, mainUri));
            return cachedDestination;
          }
        } catch (IOException e) {
          // Ignore error trying to get. We'll just download again.
        }
      }

      if (rewrittenUrls.isEmpty()) {
        StringBuilder message = new StringBuilder("Cache miss and no url specified");
        if (!originalUris.isEmpty()) {
          message.append(" - ");
          message.append(getRewriterBlockedAllUrlsMessage(originalUris));
        }
        throw new IOException(message.toString());
      }

      for (Path dir : distdir) {
        if (!dir.exists()) {
          // This is not a warning (and probably we even should drop the message); it is
          // perfectly fine to have a common rc-file pointing to a volume that is sometimes,
          // but not always mounted.
          eventHandler.handle(Event.info("non-existent distdir " + dir));
        } else if (!dir.isDirectory()) {
          eventHandler.handle(Event.warn("distdir " + dir + " is not a directory"));
        } else {
          for (String name : candidateFileNames) {
            boolean match = false;
            Path candidate = dir.getRelative(name);
            try {
              eventHandler.post(
                  new CacheProgress(
                      mainUri.toString(), "Checking " + cacheKeyType + " of " + candidate));
              match = RepositoryCache.getChecksum(cacheKeyType, candidate).equals(cacheKey);
            } catch (IOException e) {
              // Not finding anything in a distdir is a normal case, so handle it absolutely
              // quietly. In fact, it is common to specify a whole list of dist dirs,
              // with the assumption that only one will contain an entry.
            } finally {
              eventHandler.post(new CacheProgress(mainUri.toString()));
            }
            if (match) {
              if (isCachingByProvidedChecksum) {
                try {
                  repositoryCache.put(cacheKey, candidate, cacheKeyType, canonicalId);
                } catch (IOException e) {
                  eventHandler.handle(
                      Event.warn("Failed to copy " + candidate + " to repository cache: " + e));
                }
              }
              destination.getParentDirectory().createDirectoryAndParents();
              FileSystemUtils.copyFile(candidate, destination);
              return destination;
            }
          }
        }
      }
    }

    if (disableDownload) {
      throw new IOException(String.format("Failed to download %s: download is disabled.", context));
    }

    if (rewrittenUrls.isEmpty() && !originalUris.isEmpty()) {
      throw new IOException(getRewriterBlockedAllUrlsMessage(originalUris));
    }

    for (int attempt = 0; attempt <= retries; ++attempt) {
      try {
        downloader.download(
            rewrittenUrls,
            rewrittenAuthHeaders,
            checksum,
            canonicalId,
            destination,
            eventHandler,
            clientEnv,
            type);
        break;
      } catch (ContentLengthMismatchException e) {
        if (attempt == retries) {
          throw e;
        }
      } catch (InterruptedIOException e) {
        throw new InterruptedException(e.getMessage());
      }
    }

    if (isCachingByProvidedChecksum) {
      repositoryCache.put(
          checksum.get().toString(), destination, checksum.get().getKeyType(), canonicalId);
    } else if (repositoryCache.isEnabled()) {
      repositoryCache.put(destination, KeyType.SHA256, canonicalId);
    }

    return destination;
  }

  /**
   * Downloads the contents of one URL and reads it into a byte array.
   *
   * <p>If the checksum and path to the repository cache is specified, attempt to load the file from
   * the {@link RepositoryCache}. If it doesn't exist, proceed to download the file and load it into
   * the cache prior to returning the value.
   *
   * @param originalUri the original URI of the file
   * @param eventHandler CLI progress reporter
   * @param clientEnv environment variables in shell issuing this command
   * @throws IllegalArgumentException on parameter badness, which should be checked beforehand
   * @throws IOException if download was attempted and ended up failing
   * @throws InterruptedException if this thread is being cast into oblivion
   */
  public byte[] downloadAndReadOneUrl(
      URI originalUri, ExtendedEventHandler eventHandler, Map<String, String> clientEnv)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
    Map<URI, Map<String, List<String>>> authHeaders = ImmutableMap.of();
    ImmutableList<URI> rewrittenUris = ImmutableList.of(originalUri);

    if (netrcCreds != null) {
      Map<String, List<String>> metadata = netrcCreds.getRequestMetadata(originalUri);
      if (!metadata.isEmpty()) {
        Entry<String, List<String>> headers = metadata.entrySet().iterator().next();
        authHeaders =
            ImmutableMap.of(
                originalUri,
                ImmutableMap.of(headers.getKey(), ImmutableList.of(headers.getValue().get(0))));
      }
    }

    if (rewriter != null) {
      ImmutableList<UrlRewriter.RewrittenURL> rewrittenUrlMappings =
          rewriter.amend(ImmutableList.of(originalUri));
      rewrittenUris =
          rewrittenUrlMappings.stream().map(RewrittenURL::uri).collect(toImmutableList());
      authHeaders = rewriter.updateAuthHeaders(rewrittenUrlMappings, authHeaders, netrcCreds);
    }

    if (rewrittenUris.isEmpty()) {
      throw new IOException(getRewriterBlockedAllUrlsMessage(ImmutableList.of(originalUri)));
    }

    HttpDownloader httpDownloader = new HttpDownloader();
    for (int attempt = 0; attempt <= retries; ++attempt) {
      try {
        return httpDownloader.downloadAndReadOneUrl(
            rewrittenUris.get(0), authHeaders, eventHandler, clientEnv);
      } catch (ContentLengthMismatchException e) {
        if (attempt == retries) {
          throw e;
        }
      } catch (InterruptedIOException e) {
        throw new InterruptedException(e.getMessage());
      }
    }

    throw new IllegalStateException("Unexpected error: file should have been downloaded.");
  }

  @Nullable
  private String getRewriterBlockedAllUrlsMessage(List<URI> originalUris) {
    if (rewriter == null) {
      return null;
    }
    StringBuilder message = new StringBuilder("Configured URL rewriter blocked all URLs: ");
    message.append(originalUris);
    String rewriterMessage = rewriter.getAllBlockedMessage();
    if (rewriterMessage != null) {
      message.append(" - ").append(rewriterMessage);
    }
    return message.toString();
  }

  private Path getDownloadDestination(URI uri, Optional<String> type, Path output) {
    if (!type.isPresent()) {
      return output;
    }
    String basename =
        MoreObjects.firstNonNull(
            Strings.emptyToNull(PathFragment.create(uri.getPath()).getBaseName()), "temp");
    if (!type.get().isEmpty()) {
      String suffix = "." + type.get();
      if (!basename.endsWith(suffix)) {
        basename += suffix;
      }
    }
    return output.getRelative(basename);
  }

  /**
   * Deterimine the list of filenames to look for in the distdirs. Note that an output name may be
   * specified that is unrelated to the primary URL. This happens, e.g., when the paramter output is
   * specified in ctx.download.
   */
  private static ImmutableSet<String> getCandidateFileNames(URI uri, Path destination) {
    String urlBaseName = PathFragment.create(uri.getPath()).getBaseName();
    if (!Strings.isNullOrEmpty(urlBaseName) && !urlBaseName.equals(destination.getBaseName())) {
      return ImmutableSet.of(urlBaseName, destination.getBaseName());
    } else {
      return ImmutableSet.of(destination.getBaseName());
    }
  }

  private static class CacheProgress implements ExtendedEventHandler.FetchProgress {
    private final String originalUrl;
    private final String progress;
    private final boolean isFinished;

    CacheProgress(String originalUrl, String progress) {
      this.originalUrl = originalUrl;
      this.progress = progress;
      this.isFinished = false;
    }

    CacheProgress(String originalUrl) {
      this.originalUrl = originalUrl;
      this.progress = "";
      this.isFinished = true;
    }

    @Override
    public String getResourceIdentifier() {
      return originalUrl;
    }

    @Override
    public String getProgress() {
      return progress;
    }

    @Override
    public boolean isFinished() {
      return isFinished;
    }
  }
}
