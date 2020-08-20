// Copyright 2019 The Bazel Authors. All rights reserved.
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
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class UrlRewriter {

  private final Set<String> REWRITABLE_SCHEMES = ImmutableSet.of("http", "https");

  private final UrlRewriterConfig config;
  private final Function<URL, Set<URL>> rewriter;
  private final Consumer<String> log;

  public UrlRewriter(Consumer<String> log, Reader reader) {
    this.log = Preconditions.checkNotNull(log);
    Preconditions.checkNotNull(reader, "UrlRewriterConfig source must be set");
    this.config = new UrlRewriterConfig(reader);

    this.rewriter = this::rewrite;
  }

  public static UrlRewriter getDownloaderUrlRewriter(String remoteDownloaderConfig, Reporter reporter) {
    Consumer<String> log = str -> reporter.handle(Event.info(str));

    if (Strings.isNullOrEmpty(remoteDownloaderConfig)) {
      return new UrlRewriter(log, new StringReader(""));
    }

    try (BufferedReader reader = Files.newBufferedReader(Paths.get(remoteDownloaderConfig))) {
      return new UrlRewriter(log, reader);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  public List<URL> amend(List<URL> urls) {
    Objects.requireNonNull(urls, "URLS to check must be set but may be empty");

    ImmutableList<URL> rewritten = urls.stream()
      .map(rewriter)
      .flatMap(Collection::stream)
      .collect(ImmutableList.toImmutableList());

    if (!urls.equals(rewritten)) {
      log.accept(String.format("Rewritten %s as %s", urls, rewritten));
    }

    return rewritten;
  }

  private Set<URL> rewrite(URL url) {
    Preconditions.checkNotNull(url);

    // Cowardly refuse to rewrite non-HTTP(S) urls
    if (REWRITABLE_SCHEMES.stream().noneMatch(scheme -> scheme.equalsIgnoreCase(url.getProtocol()))) {
      return ImmutableSet.of(url);
    }

    Set<URL> rewrittenUrls = applyRewriteRules(url);

    ImmutableSet.Builder<URL> toReturn = ImmutableSet.builder();
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

  private boolean isMatchingHostName(URL url, String host) {
    return host.equals(url.getHost()) || url.getHost().endsWith("." + host);
  }

  private Set<URL> applyRewriteRules(URL url) {
    String withoutScheme = url.toString().substring(url.getProtocol().length() + 3);

    ImmutableSet.Builder<URL> rewrittenUrls = ImmutableSet.builder();

    boolean matchMade = false;
    for (Map.Entry<Pattern, Collection<String>> entry : config.getRewrites().entrySet()) {
      Matcher matcher = entry.getKey().matcher(withoutScheme);
      if (matcher.matches()) {
        matchMade = true;

        for (String replacement : entry.getValue()) {
          try {
            rewrittenUrls.add(new URL(url.getProtocol() + "://" + matcher.replaceFirst(replacement)));
          } catch (MalformedURLException e) {
            throw new IllegalStateException(e);
          }
        }
      }
    }
    if (!matchMade) {
      rewrittenUrls.add(url);
    }

    return rewrittenUrls.build();
  }
}
