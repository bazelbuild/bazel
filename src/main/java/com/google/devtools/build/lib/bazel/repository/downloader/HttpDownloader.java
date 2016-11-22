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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache;
import com.google.devtools.build.lib.bazel.repository.cache.RepositoryCache.KeyType;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Helper class for downloading a file from a URL.
 */
public class HttpDownloader {
  private static final int BUFFER_SIZE = 32 * 1024;
  private static final int KB = 1024;
  private static final String UNITS = " KMGTPEY";
  private static final double LOG_OF_KB = Math.log(1024);

  private final ScheduledExecutorService scheduler;
  private Location ruleUrlAttributeLocation;

  protected final RepositoryCache repositoryCache;

  public HttpDownloader(RepositoryCache repositoryCache) {
    this.scheduler = Executors.newScheduledThreadPool(1);
    this.ruleUrlAttributeLocation = null;
    this.repositoryCache = repositoryCache;
  }

  public Path download(
      Rule rule, Path outputDirectory, EventHandler eventHandler, Map<String, String> clientEnv)
      throws RepositoryFunctionException, InterruptedException {
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    String url;
    String sha256;
    String type;
    try {
      ruleUrlAttributeLocation = rule.getAttributeLocation("url");

      url = mapper.get("url", Type.STRING);
      sha256 = mapper.get("sha256", Type.STRING);
      type = mapper.isAttributeValueExplicitlySpecified("type")
          ? mapper.get("type", Type.STRING) : "";
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }

    try {
      return download(url, sha256, type, outputDirectory, eventHandler, clientEnv);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              "Error downloading from " + url + " to " + outputDirectory + ": " + e.getMessage()),
          SkyFunctionException.Transience.TRANSIENT);
    }
  }

  /**
   * Attempt to download a file from the repository's URL. Returns the path to the file downloaded.
   *
   * If the SHA256 checksum and path to the repository cache is specified, attempt
   * to load the file from the RepositoryCache. If it doesn't exist, proceed to
   * download the file and load it into the cache prior to returning the value.
   */
  public Path download(
      String urlString, String sha256, String type, Path outputDirectory,
      EventHandler eventHandler, Map<String, String> clientEnv)
          throws IOException, InterruptedException, RepositoryFunctionException {
    Path destination = getDownloadDestination(urlString, type, outputDirectory);

    // Used to decide whether to cache the download at the end of this method.
    boolean isCaching = false;

    if (RepositoryCache.KeyType.SHA256.isValid(sha256)) {
      try {
        String currentSha256 = RepositoryCache.getChecksum(KeyType.SHA256, destination);
        if (currentSha256.equals(sha256)) {
          // No need to download.
          return destination;
        }
      } catch (IOException e) {
        // Ignore error trying to hash. We'll attempt to retrieve from cache or just download again.
      }

      if (repositoryCache.isEnabled()) {
        isCaching = true;

        Path cachedDestination = repositoryCache.get(sha256, destination, KeyType.SHA256);
        if (cachedDestination != null) {
          // Cache hit!
          return cachedDestination;
        }
      }
    }

    AtomicInteger totalBytes = new AtomicInteger(0);
    final ScheduledFuture<?> loggerHandle = getLoggerHandle(totalBytes, eventHandler, urlString);
    final URL url = new URL(urlString);

    try (OutputStream out = destination.getOutputStream();
         HttpConnection connection = HttpConnection.createAndConnect(url, clientEnv)) {
      InputStream inputStream = connection.getInputStream();
      int read;
      byte[] buf = new byte[BUFFER_SIZE];
      while ((read = inputStream.read(buf)) > 0) {
        totalBytes.addAndGet(read);
        out.write(buf, 0, read);
        if (Thread.interrupted()) {
          throw new InterruptedException("Download interrupted");
        }
      }
      if (connection.getContentLength() != -1
          && totalBytes.get() != connection.getContentLength()) {
        throw new IOException("Expected " + formatSize(connection.getContentLength()) + ", got "
            + formatSize(totalBytes.get()));
      }
    } catch (IOException e) {
      throw new IOException(
          "Error downloading " + url + " to " + destination + ": " + e.getMessage());
    } finally {
      scheduler.schedule(new Runnable() {
        @Override
        public void run() {
          loggerHandle.cancel(true);
        }
      }, 0, TimeUnit.SECONDS);
    }

    if (!sha256.isEmpty()) {
      RepositoryCache.assertFileChecksum(sha256, destination, KeyType.SHA256);
    }

    if (isCaching) {
      repositoryCache.put(sha256, destination, KeyType.SHA256);
    }

    return destination;
  }

  private Path getDownloadDestination(String urlString, String type, Path outputDirectory)
      throws RepositoryFunctionException {
    URI uri = null;
    try {
      uri = new URI(urlString);
    } catch (URISyntaxException e) {
      throw new RepositoryFunctionException(
          new EvalException(ruleUrlAttributeLocation, e), Transience.PERSISTENT);
    }
    if (type == null) {
      return outputDirectory;
    } else {
      String filename = new PathFragment(uri.getPath()).getBaseName();
      if (filename.isEmpty()) {
        filename = "temp";
      } else if (!type.isEmpty()) {
        filename += "." + type;
      }
      return outputDirectory.getRelative(filename);
    }
  }

  private ScheduledFuture<?> getLoggerHandle(
      final AtomicInteger totalBytes, final EventHandler eventHandler, final String urlString) {
    final Runnable logger = new Runnable() {
      @Override
      public void run() {
        try {
          eventHandler.handle(Event.progress(
              "Downloading from " + urlString + ": " + formatSize(totalBytes.get())));
        } catch (Exception e) {
          eventHandler.handle(Event.error(
              "Error generating download progress: " + e.getMessage()));
        }
      }
    };
    return scheduler.scheduleAtFixedRate(logger, 0, 1, TimeUnit.SECONDS);
  }

  private String formatSize(int bytes) {
    if (bytes < KB) {
      return bytes + "B";
    }
    int logBaseUnitOfBytes = (int) (Math.log(bytes) / LOG_OF_KB);
    if (logBaseUnitOfBytes < 0 || logBaseUnitOfBytes >= UNITS.length()) {
      return bytes + "B";
    }
    return (int) (bytes / Math.pow(KB, logBaseUnitOfBytes))
        + (UNITS.charAt(logBaseUnitOfBytes) + "B");
  }

}
