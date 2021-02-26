// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.UncheckedIOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Helper class for taking URLs and converting them according to an optional config specified by
 * {@link com.google.devtools.build.lib.bazel.repository.RepositoryOptions#downloaderConfig}.
 *
 * <p>The primary reason for doing this is to allow a bazel user to redirect particular URLs to
 * (eg.) local mirrors without needing to rewrite third party rulesets.
 */
public class UrlRewriter {

  private static final ImmutableSet<String> REWRITABLE_SCHEMES = ImmutableSet.of("http", "https");

  private final UrlRewriterConfig config;
  private final Function<URL, List<URL>> rewriter;
  private final Consumer<String> log;

  @VisibleForTesting
  UrlRewriter(Consumer<String> log, String filePathForErrorReporting, Reader reader)
      throws UrlRewriterParseException {
    this.log = Preconditions.checkNotNull(log);
    Preconditions.checkNotNull(reader, "UrlRewriterConfig source must be set");
    this.config = new UrlRewriterConfig(filePathForErrorReporting, reader);

    this.rewriter = this::rewrite;
  }

  /**
   * Obtain a new {@code UrlRewriter} configured with the specified config file.
   *
   * @param configPath Path to the config file to use. May be null.
   * @param reporter Used for logging when URLs are rewritten.
   */
  public static UrlRewriter getDownloaderUrlRewriter(String configPath, Reporter reporter)
      throws UrlRewriterParseException {
    Consumer<String> log = str -> reporter.handle(Event.info(str));

    if (Strings.isNullOrEmpty(configPath)) {
      return new UrlRewriter(log, "", new StringReader(""));
    }

    try (BufferedReader reader = Files.newBufferedReader(Paths.get(configPath))) {
      return new UrlRewriter(log, configPath, reader);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  /**
   * Rewrites {@code urls} using the configuration provided to {@link
   * #getDownloaderUrlRewriter(String, Reporter)}. The returned list of URLs may be empty if the
   * configuration used blocks all the input URLs.
   *
   * @param urls The input list of {@link URL}s. May be empty.
   * @return The amended lists of URLs.
   */
  public List<URL> amend(List<URL> urls) {
    Objects.requireNonNull(urls, "URLS to check must be set but may be empty");

    ImmutableList<URL> rewritten =
        urls.stream().map(rewriter).flatMap(Collection::stream).collect(toImmutableList());

    if (!urls.equals(rewritten)) {
      log.accept(String.format("Rewritten %s as %s", urls, rewritten));
    }

    return rewritten;
  }

  private ImmutableList<URL> rewrite(URL url) {
    Preconditions.checkNotNull(url);

    // Cowardly refuse to rewrite non-HTTP(S) urls
    if (REWRITABLE_SCHEMES.stream()
        .noneMatch(scheme -> Ascii.equalsIgnoreCase(scheme, url.getProtocol()))) {
      return ImmutableList.of(url);
    }

    List<URL> rewrittenUrls = applyRewriteRules(url);

    ImmutableList.Builder<URL> toReturn = ImmutableList.builder();
    // Now iterate over the URLs
    for (URL consider : rewrittenUrls) {
      // If there's an allow entry, add it to the set to return and continue
      if (isAllowMatched(consider)) {
        toReturn.add(consider);
        continue;
      }

      // If there's no block that matches the domain, add it to the set to return and continue
      if (!isBlockMatched(consider)) {
        toReturn.add(consider);
      }
    }

    return toReturn.build();
  }

  private boolean isAllowMatched(URL url) {
    for (String host : config.getAllowList()) {
      if (isMatchingHostName(url, host)) {
        return true;
      }
    }
    return false;
  }

  private boolean isBlockMatched(URL url) {
    for (String host : config.getBlockList()) {
      // Allow a wild-card block
      if ("*".equals(host)) {
        return true;
      }

      if (isMatchingHostName(url, host)) {
        return true;
      }
    }
    return false;
  }

  private static boolean isMatchingHostName(URL url, String host) {
    return host.equals(url.getHost()) || url.getHost().endsWith("." + host);
  }

  private ImmutableList<URL> applyRewriteRules(URL url) {
    String withoutScheme = url.toString().substring(url.getProtocol().length() + 3);

    ImmutableSet.Builder<String> rewrittenUrls = ImmutableSet.builder();

    boolean matchMade = false;
    for (Map.Entry<Pattern, Collection<String>> entry : config.getRewrites().entrySet()) {
      Matcher matcher = entry.getKey().matcher(withoutScheme);
      if (matcher.matches()) {
        matchMade = true;

        for (String replacement : entry.getValue()) {
          rewrittenUrls.add(matcher.replaceFirst(replacement));
        }
      }
    }

    if (!matchMade) {
      return ImmutableList.of(url);
    }

    return rewrittenUrls.build().stream()
        .map(
            urlString -> {
              try {
                return new URL(url.getProtocol() + "://" + urlString);
              } catch (MalformedURLException e) {
                throw new IllegalStateException(e);
              }
            })
        .collect(toImmutableList());
  }

  @Nullable
  public String getAllBlockedMessage() {
    return config.getAllBlockedMessage();
  }
}
