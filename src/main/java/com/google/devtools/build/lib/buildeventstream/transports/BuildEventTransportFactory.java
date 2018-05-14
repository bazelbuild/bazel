// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream.transports;

import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

/** Factory used to create a Set of BuildEventTransports from BuildEventStreamOptions. */
public enum BuildEventTransportFactory {
  TEXT_TRANSPORT {
    @Override
    protected boolean enabled(BuildEventStreamOptions options) {
      return !isNullOrEmpty(options.getBuildEventTextFile());
    }

    @Override
    protected BuildEventTransport create(BuildEventStreamOptions options,
        PathConverter pathConverter) throws IOException {
      return new TextFormatFileTransport(
          options.getBuildEventTextFile(),
          options.getBuildEventTextFilePathConversion() ? pathConverter : new NullPathConverter());
    }
  },

  BINARY_TRANSPORT {
    @Override
    protected boolean enabled(BuildEventStreamOptions options) {
      return !isNullOrEmpty(options.getBuildEventBinaryFile());
    }

    @Override
    protected BuildEventTransport create(BuildEventStreamOptions options,
        PathConverter pathConverter) throws IOException {
      return new BinaryFormatFileTransport(
          options.getBuildEventBinaryFile(),
          options.getBuildEventBinaryFilePathConversion()
              ? pathConverter
              : new NullPathConverter());
    }
  },

  JSON_TRANSPORT {
    @Override
    protected boolean enabled(BuildEventStreamOptions options) {
      return !isNullOrEmpty(options.getBuildEventJsonFile());
    }

    @Override
    protected BuildEventTransport create(
        BuildEventStreamOptions options, PathConverter pathConverter) throws IOException {
      return new JsonFormatFileTransport(
          options.getBuildEventJsonFile(),
          options.getBuildEventJsonFilePathConversion() ? pathConverter : new NullPathConverter());
    }
  };

  /**
   * Creates a {@link ImmutableSet} of {@link BuildEventTransport} based on the specified {@link
   * BuildEventStreamOptions}.
   *
   * @param options Options used configure and create the returned BuildEventTransports.
   * @return A {@link ImmutableSet} of BuildEventTransports. This set may be empty.
   * @throws IOException Exception propagated from a {@link BuildEventTransport} creation failure.
   */
  public static ImmutableSet<BuildEventTransport> createFromOptions(BuildEventStreamOptions options,
      PathConverter pathConverter) throws IOException {
    ImmutableSet.Builder<BuildEventTransport> buildEventTransportsBuilder = ImmutableSet.builder();
    for (BuildEventTransportFactory transportFactory : BuildEventTransportFactory.values()) {
      if (transportFactory.enabled(options)) {
        buildEventTransportsBuilder.add(transportFactory.create(options, pathConverter));
      }
    }
    return buildEventTransportsBuilder.build();
  }

  /** Returns true if this factory BuildEventTransport is enabled by the specified options. */
  protected abstract boolean enabled(BuildEventStreamOptions options);

  /** Creates a BuildEventTransport from the specified options. */
  protected abstract BuildEventTransport create(BuildEventStreamOptions options,
      PathConverter pathConverter) throws IOException;

  private static class NullPathConverter implements PathConverter {
    @Override
    public String apply(Path path) {
      return pathToUriString(path.getPathString());
    }
  }

  /**
   * Returns the path encoded as an {@link URI}.
   *
   * <p>This concrete implementation returns URIs with "file" as the scheme. For Example: - On Unix
   * the path "/tmp/foo bar.txt" will be encoded as "file:///tmp/foo%20bar.txt". - On Windows the
   * path "C:\Temp\Foo Bar.txt" will be encoded as "file:///C:/Temp/Foo%20Bar.txt"
   *
   * <p>Implementors extending this class for special filesystems will likely need to override this
   * method.
   *
   * @throws URISyntaxException if the URI cannot be constructed.
   */
  static String pathToUriString(String path) {
    if (!path.startsWith("/")) {
      // On Windows URI's need to start with a '/'. i.e. C:\Foo\Bar would be file:///C:/Foo/Bar
      path = "/" + path;
    }
    try {
      return new URI(
              "file",
              // Needs to be "" instead of null, so that toString() will append "//" after the
              // scheme.
              // We need this for backwards compatibility reasons as some consumers of the BEP are
              // broken.
              "",
              path,
              null,
              null)
          .toString();
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }
}
