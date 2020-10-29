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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Models the downloader config file. This file has a line-based format, with each line starting
 * with a directive and then the action to take. These directives are:
 *
 * <ul>
 *   <li>{@code allow hostName} Will allow access to the given host and subdomains
 *   <li>{@code block hostName} Will block access to the given host and subdomains
 *   <li>{@code rewrite pattern pattern} Rewrite a URL using the given pattern. Back references are
 *       numbered from `$1`
 * </ul>
 *
 * The directives are applied in the order `rewrite, allow, block'. An example config may look like:
 *
 * <pre>
 *     block mvnrepository.com
 *     block maven-central.storage.googleapis.com
 *     block gitblit.github.io
 *     rewrite repo.maven.apache.org/maven2/(.*) artifacts.mycorp.com/libs-release/$1
 *
 *     block github.com
 *     rewrite github.com/([^/]+)/([^/]+)/releases/download/([^/]+)/(.*) \
 *             artifacts.mycorp.com/github-releases/$1/$2/releases/download/$3/$4
 *     rewrite github.com/([^/]+)/([^/]+)/archive/(.+).(tar.gz|zip) \
 *             artifacts.mycorp.com/api/vcs/downloadRelease/github.com/$1/$2/$3?ext=$4
 * </pre>
 *
 * In addition, you can block all hosts using the @{code *} wildcard.
 *
 * <p>Comments within the config file are allowed, and must be on their own line preceded by a
 * {@code #}.
 */
class UrlRewriterConfig {

  private static final Splitter SPLITTER =
      Splitter.onPattern("\\s+").omitEmptyStrings().trimResults();

  // A set of domain names that should be accessible.
  private final Set<String> allowList;
  // A set of domain names that should be blocked.
  private final Set<String> blockList;
  // A set of patterns matching "everything in the url after the scheme" to rewrite rules.
  private final ImmutableMultimap<Pattern, String> rewrites;

  /**
   * Constructor to use. The {@code config} will be read to completion.
   *
   * @throws UncheckedIOException If any processing problems occur.
   */
  public UrlRewriterConfig(Reader config) {
    ImmutableSet.Builder<String> allowList = ImmutableSet.builder();
    ImmutableSet.Builder<String> blockList = ImmutableSet.builder();
    ImmutableMultimap.Builder<Pattern, String> rewrites = ImmutableMultimap.builder();

    try (BufferedReader reader = new BufferedReader(config)) {
      for (String line = reader.readLine(); line != null; line = reader.readLine()) {
        // Find the first word
        List<String> parts = SPLITTER.splitToList(line);
        if (parts.isEmpty()) {
          continue;
        }

        // Allow comments to use #
        if (parts.get(0).startsWith("#")) {
          continue;
        }

        switch (parts.get(0)) {
          case "allow":
            if (parts.size() != 2) {
              throw new IllegalStateException(
                  "Only the host name is allowed after `allow`: " + line);
            }
            allowList.add(parts.get(1));
            break;

          case "block":
            if (parts.size() != 2) {
              throw new IllegalStateException(
                  "Only the host name is allowed after `block`: " + line);
            }
            blockList.add(parts.get(1));
            break;

          case "rewrite":
            if (parts.size() != 3) {
              throw new IllegalStateException(
                  "Only the matching pattern and rewrite pattern is allowed after `rewrite`: "
                      + line);
            }
            rewrites.put(Pattern.compile(parts.get(1)), parts.get(2));
            break;

          default:
            throw new IllegalStateException("Unable to parse: " + line);
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }

    this.allowList = allowList.build();
    this.blockList = blockList.build();
    this.rewrites = rewrites.build();
  }

  /** Returns all {@code allow} directives. */
  public Set<String> getAllowList() {
    return allowList;
  }

  /** Returns all {@code block} directives. */
  public Set<String> getBlockList() {
    return blockList;
  }

  /**
   * Returns a {@link Map} of {@link Pattern} to match against, and the rewriting changes to apply
   * when matched.
   */
  public Map<Pattern, Collection<String>> getRewrites() {
    return rewrites.asMap();
  }
}
