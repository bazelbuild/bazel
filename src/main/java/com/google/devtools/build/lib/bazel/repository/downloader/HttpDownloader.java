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

import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCacheHitEvent;
import com.google.devtools.build.lib.buildeventstream.FetchEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.JavaSleeper;
import com.google.devtools.build.lib.util.Sleeper;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Semaphore;

/**
 * Bazel file downloader.
 *
 * <p>This class uses {@link HttpConnectorMultiplexer} to connect to HTTP mirrors and then reads the
 * file to disk.
 */
public class HttpDownloader extends Downloader {

  private static final int MAX_PARALLEL_DOWNLOADS = 8;
  private static final Semaphore semaphore = new Semaphore(MAX_PARALLEL_DOWNLOADS, true);

  protected final RepositoryCache repositoryCache;
  private List<Path> distdir = ImmutableList.of();
  private float timeoutScaling = 1.0f;

  public HttpDownloader(RepositoryCache repositoryCache) {
    this.repositoryCache = repositoryCache;
  }

  public void setDistdir(List<Path> distdir) {
    this.distdir = ImmutableList.copyOf(distdir);
  }

  public void setTimeoutScaling(float timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  @Override
  public Path download(
      List<URL> urls,
      Map<URI, Map<String, String>> authHeaders,
      Optional<Checksum> checksum,
      String canonicalId,
      Optional<String> type,
      Path output,
      ExtendedEventHandler eventHandler,
      Map<String, String> clientEnv,
      String repo)
      throws IOException, InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }

    URL mainUrl; // The "main" URL for this request
    // Used for reporting only and determining the file name only.
    if (urls.isEmpty()) {
      if (type.isPresent() && !Strings.isNullOrEmpty(type.get())) {
        mainUrl = new URL("http://nonexistent.example.org/cacheprobe." + type.get());
      } else {
        mainUrl = new URL("http://nonexistent.example.org/cacheprobe");
      }
    } else {
      mainUrl = urls.get(0);
    }
    Path destination = getDownloadDestination(mainUrl, type, output);

    // Is set to true if the value should be cached by the checksum value provided
    boolean isCachingByProvidedChecksum = false;

    if (checksum.isPresent()) {
      String cacheKey = checksum.get().toString();
      KeyType cacheKeyType = checksum.get().getKeyType();
      try {
        String currentChecksum = RepositoryCache.getChecksum(cacheKeyType, destination);
        if (currentChecksum.equals(cacheKey)) {
          // No need to download.
          return destination;
        }
      } catch (IOException e) {
        // Ignore error trying to hash. We'll attempt to retrieve from cache or just download again.
      }

      if (repositoryCache.isEnabled()) {
        isCachingByProvidedChecksum = true;

        try {
          Path cachedDestination =
              repositoryCache.get(cacheKey, destination, cacheKeyType, canonicalId);
          if (cachedDestination != null) {
            // Cache hit!
            eventHandler.post(new RepositoryCacheHitEvent(repo, cacheKey, mainUrl));
            return cachedDestination;
          }
        } catch (IOException e) {
          // Ignore error trying to get. We'll just download again.
        }
      }

      if (urls.isEmpty()) {
        throw new IOException("Cache miss and no url specified");
      }

      for (Path dir : distdir) {
        if (!dir.exists()) {
          // This is not a warning (and probably we even should drop the message); it is
          // perfectly fine to have a common rc-file pointing to a volume that is sometimes,
          // but not always mounted.
          eventHandler.handle(Event.info("non-existent distir " + dir));
        } else if (!dir.isDirectory()) {
          eventHandler.handle(Event.warn("distdir " + dir + " is not a directory"));
        } else {
          boolean match = false;
          Path candidate = dir.getRelative(destination.getBaseName());
          try {
            match = RepositoryCache.getChecksum(cacheKeyType, candidate).equals(cacheKey);
          } catch (IOException e) {
            // Not finding anything in a distdir is a normal case, so handle it absolutely
            // quietly. In fact, it is not uncommon to specify a whole list of dist dirs,
            // with the asumption that only one will contain an entry.
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
            FileSystemUtils.createDirectoryAndParents(destination.getParentDirectory());
            FileSystemUtils.copyFile(candidate, destination);
            return destination;
          }
        }
      }
    }

    Clock clock = new JavaClock();
    Sleeper sleeper = new JavaSleeper();
    Locale locale = Locale.getDefault();
    ProxyHelper proxyHelper = new ProxyHelper(clientEnv);
    HttpConnector connector =
        new HttpConnector(locale, eventHandler, proxyHelper, sleeper, timeoutScaling);
    ProgressInputStream.Factory progressInputStreamFactory =
        new ProgressInputStream.Factory(locale, clock, eventHandler);
    HttpStream.Factory httpStreamFactory = new HttpStream.Factory(progressInputStreamFactory);
    HttpConnectorMultiplexer multiplexer =
        new HttpConnectorMultiplexer(eventHandler, connector, httpStreamFactory, clock, sleeper);

    // Connect to the best mirror and download the file, while reporting progress to the CLI.
    semaphore.acquire();
    boolean success = false;
    try (HttpStream payload = multiplexer.connect(urls, checksum, authHeaders);
        OutputStream out = destination.getOutputStream()) {
      ByteStreams.copy(payload, out);
      success = true;
    } catch (InterruptedIOException e) {
      throw new InterruptedException();
    } catch (IOException e) {
      throw new IOException(
          "Error downloading " + urls + " to " + destination + ": " + e.getMessage());
    } finally {
      semaphore.release();
      eventHandler.post(new FetchEvent(urls.get(0).toString(), success));
    }

    if (isCachingByProvidedChecksum) {
      repositoryCache.put(
          checksum.get().toString(), destination, checksum.get().getKeyType(), canonicalId);
    } else if (repositoryCache.isEnabled()) {
      String newSha256 = repositoryCache.put(destination, KeyType.SHA256, canonicalId);
      eventHandler.handle(Event.info("SHA256 (" + urls.get(0) + ") = " + newSha256));
    }

    return destination;
  }
}
