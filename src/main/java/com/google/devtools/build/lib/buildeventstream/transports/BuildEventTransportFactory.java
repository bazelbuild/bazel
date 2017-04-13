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
import com.google.common.collect.ImmutableSet.Builder;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

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
    Builder<BuildEventTransport> buildEventTransportsBuilder = ImmutableSet.builder();
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
      return "file://" + path;
    }
  }
}
