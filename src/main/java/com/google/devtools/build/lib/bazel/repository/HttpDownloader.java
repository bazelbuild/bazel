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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Authenticator;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.PasswordAuthentication;
import java.net.Proxy;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.Nullable;

/**
 * Helper class for downloading a file from a URL.
 */
public class HttpDownloader {
  private static final int BUFFER_SIZE = 32 * 1024;
  private static final int KB = 1024;
  private static final String UNITS = " KMGTPEY";
  private static final double LOG_OF_KB = Math.log(1024);

  private final String urlString;
  private final String sha256;
  private final String type;
  private final Path outputDirectory;
  private final EventHandler eventHandler;
  private final ScheduledExecutorService scheduler;

  private HttpDownloader(EventHandler eventHandler, String urlString, String sha256,
      Path outputDirectory, String type) {
    this.urlString = urlString;
    this.sha256 = sha256;
    this.outputDirectory = outputDirectory;
    this.eventHandler = eventHandler;
    this.scheduler = Executors.newScheduledThreadPool(1);
    this.type = type;
  }

  @Nullable
  public static Path download(Rule rule, Path outputDirectory, EventHandler eventHandler)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    String url = mapper.get("url", Type.STRING);
    String sha256 = mapper.get("sha256", Type.STRING);
    String type = mapper.has("type", Type.STRING) ? mapper.get("type", Type.STRING) : "";

    try {
      return new HttpDownloader(eventHandler, url, sha256, outputDirectory, type).download();
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException("Error downloading from "
          + url + " to " + outputDirectory + ": " + e.getMessage()),
          SkyFunctionException.Transience.TRANSIENT);
    }
  }

  /**
   * Attempt to download a file from the repository's URL. Returns the path to the file downloaded.
   */
  public Path download() throws IOException {
    URL url = new URL(urlString);
    String filename = new PathFragment(url.getPath()).getBaseName();
    if (filename.isEmpty()) {
      filename = "temp";
    }
    if (!type.isEmpty()) {
      filename += "." + type;
    }
    Path destination = outputDirectory.getRelative(filename);

    if (!sha256.isEmpty()) {
      try {
        String currentSha256 = getHash(Hashing.sha256().newHasher(), destination);
        if (currentSha256.equals(sha256)) {
          // No need to download.
          return destination;
        }
      } catch (IOException e) {
        // Ignore error trying to hash. We'll just download again.
      }
    }

    AtomicInteger totalBytes = new AtomicInteger(0);
    final ScheduledFuture<?> loggerHandle = getLoggerHandle(totalBytes);

    try (OutputStream out = destination.getOutputStream();
         HttpConnection connection = HttpConnection.createAndConnect(url)) {
      InputStream inputStream = connection.getInputStream();
      int read;
      byte[] buf = new byte[BUFFER_SIZE];
      while ((read = inputStream.read(buf)) > 0) {
        totalBytes.addAndGet(read);
        out.write(buf, 0, read);
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

    compareHashes(destination);
    return destination;
  }

  private void compareHashes(Path destination) throws IOException {
    if (sha256.isEmpty()) {
      return;
    }
    String downloadedSha256;
    try {
      downloadedSha256 = getHash(Hashing.sha256().newHasher(), destination);
    } catch (IOException e) {
      throw new IOException(
          "Could not hash file " + destination + ": " + e.getMessage() + ", expected SHA-256 of "
              + sha256 + ")");
    }
    if (!downloadedSha256.equals(sha256)) {
      throw new IOException(
          "Downloaded file at " + destination + " has SHA-256 of " + downloadedSha256
              + ", does not match expected SHA-256 (" + sha256 + ")");
    }
  }

  private ScheduledFuture<?> getLoggerHandle(final AtomicInteger totalBytes) {
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

  private static String formatSize(int bytes) {
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

  private static class HttpConnection implements Closeable {
    private final InputStream inputStream;
    private final int contentLength;

    private HttpConnection(InputStream inputStream, int contentLength) {
      this.inputStream = inputStream;
      this.contentLength = contentLength;
    }

    public InputStream getInputStream() {
      return inputStream;
    }

    /**
     * @return The length of the response, or -1 if unknown.
     */
    public int getContentLength() {
      return contentLength;
    }

    @Override
    public void close() throws IOException {
      inputStream.close();
    }

    private static int parseContentLength(HttpURLConnection connection) {
      String length;
      try {
        length = connection.getHeaderField("Content-Length");
        if (length == null) {
          return -1;
        }
        return Integer.parseInt(length);
      } catch (NumberFormatException e) {
        return -1;
      }
    }

    public static HttpConnection createAndConnect(URL url) throws IOException {
      HttpURLConnection connection = (HttpURLConnection) url.openConnection(
          createProxyIfNeeded(url.getProtocol()));
      connection.setInstanceFollowRedirects(true);
      connection.connect();
      if (connection.getResponseCode() != HttpURLConnection.HTTP_OK) {
        InputStream errorStream = connection.getErrorStream();
        throw new IOException(connection.getResponseCode() + ": "
            + new String(ByteStreams.toByteArray(errorStream), StandardCharsets.UTF_8));
      }
      return new HttpConnection(connection.getInputStream(), parseContentLength(connection));
    }
  }

  private static Proxy createProxyIfNeeded(String protocol) throws IOException {
    if (protocol.equals("https")) {
      return createProxy(System.getenv("HTTPS_PROXY"));
    } else if (protocol.equals("http")) {
      return createProxy(System.getenv("HTTP_PROXY"));
    }
    return Proxy.NO_PROXY;
  }

  @VisibleForTesting
  static Proxy createProxy(String proxyAddress) throws IOException {
    if (Strings.isNullOrEmpty(proxyAddress)) {
      return Proxy.NO_PROXY;
    }

    // Here there be dragons.
    Pattern urlPattern =
        Pattern.compile("^(https?)://(?:([^:@]+?)(?::([^@]+?))?@)?(?:[^:]+)(?::(\\d+))?$");
    Matcher matcher = urlPattern.matcher(proxyAddress);
    if (!matcher.matches()) {
      throw new IOException("Proxy address " + proxyAddress + " is not a valid URL");
    }

    String protocol = matcher.group(1);
    final String username = matcher.group(2);
    final String password = matcher.group(3);
    String port = matcher.group(4);

    boolean https;
    switch (protocol) {
      case "https":
        https = true;
        break;
      case "http":
        https = false;
        break;
      default:
        throw new IOException("Invalid proxy protocol for " + proxyAddress);
    }

    if (username != null) {
      if (password == null) {
        throw new IOException("No password given for proxy " + proxyAddress);
      }
      System.setProperty(protocol + ".proxyUser", username);
      System.setProperty(protocol + ".proxyPassword", password);

      Authenticator.setDefault(
          new Authenticator() {
            @Override
            public PasswordAuthentication getPasswordAuthentication() {
              return new PasswordAuthentication(username, password.toCharArray());
            }
          });
    }

    if (port == null) {
      return new Proxy(Proxy.Type.HTTP, new InetSocketAddress(proxyAddress, https ? 443 : 80));
    }

    try {
      return new Proxy(
          Proxy.Type.HTTP,
          new InetSocketAddress(
              proxyAddress.substring(0, proxyAddress.lastIndexOf(':')), Integer.parseInt(port)));
    } catch (NumberFormatException e) {
      throw new IOException("Error parsing proxy port: " + proxyAddress);
    }
  }

  public static String getHash(Hasher hasher, Path path) throws IOException {
    byte byteBuffer[] = new byte[BUFFER_SIZE];
    try (InputStream stream = path.getInputStream()) {
      int numBytesRead = stream.read(byteBuffer);
      while (numBytesRead != -1) {
        if (numBytesRead != 0) {
          // If more than 0 bytes were read, add them to the hash.
          hasher.putBytes(byteBuffer, 0, numBytesRead);
        }
        numBytesRead = stream.read(byteBuffer);
      }
    }
    return hasher.hash().toString();
  }
}
