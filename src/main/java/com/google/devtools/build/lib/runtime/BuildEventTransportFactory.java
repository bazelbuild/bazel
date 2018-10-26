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

package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.isNullOrEmpty;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.LocalFilesArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.transports.BinaryFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildeventstream.transports.JsonFormatFileTransport;
import com.google.devtools.build.lib.buildeventstream.transports.TextFormatFileTransport;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.io.IOException;
import java.util.function.Consumer;

/** Factory used to create a Set of BuildEventTransports from BuildEventStreamOptions. */
public enum BuildEventTransportFactory {
  TEXT_TRANSPORT {
    @Override
    protected boolean enabled(BuildEventStreamOptions options) {
      return !isNullOrEmpty(options.getBuildEventTextFile());
    }

    @Override
    protected BuildEventTransport create(
        BuildEventStreamOptions options,
        BuildEventProtocolOptions protocolOptions,
        BuildEventArtifactUploader uploader,
        Consumer<AbruptExitException> exitFunc)
        throws IOException {
      return new TextFormatFileTransport(
          options.getBuildEventTextFile(), protocolOptions, uploader, exitFunc);
    }

    @Override
    protected boolean usePathConverter(BuildEventStreamOptions options) {
      return options.getBuildEventTextFilePathConversion();
    }
  },

  BINARY_TRANSPORT {
    @Override
    protected boolean enabled(BuildEventStreamOptions options) {
      return !isNullOrEmpty(options.getBuildEventBinaryFile());
    }

    @Override
    protected BuildEventTransport create(
        BuildEventStreamOptions options,
        BuildEventProtocolOptions protocolOptions,
        BuildEventArtifactUploader uploader,
        Consumer<AbruptExitException> exitFunc)
        throws IOException {
      return new BinaryFormatFileTransport(
          options.getBuildEventBinaryFile(), protocolOptions, uploader, exitFunc);
    }

    @Override
    protected boolean usePathConverter(BuildEventStreamOptions options) {
      return options.getBuildEventBinaryFilePathConversion();
    }
  },

  JSON_TRANSPORT {
    @Override
    protected boolean enabled(BuildEventStreamOptions options) {
      return !isNullOrEmpty(options.getBuildEventJsonFile());
    }

    @Override
    protected BuildEventTransport create(
        BuildEventStreamOptions options,
        BuildEventProtocolOptions protocolOptions,
        BuildEventArtifactUploader uploader,
        Consumer<AbruptExitException> exitFunc)
        throws IOException {
      return new JsonFormatFileTransport(
          options.getBuildEventJsonFile(), protocolOptions, uploader, exitFunc);
    }

    @Override
    protected boolean usePathConverter(BuildEventStreamOptions options) {
      return options.getBuildEventJsonFilePathConversion();
    }
  };

  /**
   * Creates a {@link ImmutableSet} of {@link BuildEventTransport} based on the specified {@link
   * BuildEventStreamOptions}.
   *
   * @return A {@link ImmutableSet} of BuildEventTransports. This set may be empty.
   * @throws IOException Exception propagated from a {@link BuildEventTransport} creation failure.
   */
  public static ImmutableSet<BuildEventTransport> createFromOptions(
      CommandEnvironment env, Consumer<AbruptExitException> exitFunc) throws IOException {
    BuildEventStreamOptions bepOptions =
        checkNotNull(
            env.getOptions().getOptions(BuildEventStreamOptions.class),
            "Could not get BuildEventStreamOptions.");
    BuildEventProtocolOptions protocolOptions =
        checkNotNull(
            env.getOptions().getOptions(BuildEventProtocolOptions.class),
            "Could not get BuildEventProtocolOptions.");
    ImmutableSet.Builder<BuildEventTransport> buildEventTransportsBuilder = ImmutableSet.builder();
    for (BuildEventTransportFactory transportFactory : BuildEventTransportFactory.values()) {
      if (transportFactory.enabled(bepOptions)) {
        BuildEventArtifactUploader uploader =
            transportFactory.usePathConverter(bepOptions)
                ? env.getRuntime()
                    .getBuildEventArtifactUploaderFactoryMap()
                    .select(protocolOptions.buildEventUploadStrategy)
                    .create(env)
                : new LocalFilesArtifactUploader();
        buildEventTransportsBuilder.add(
            transportFactory.create(bepOptions, protocolOptions, uploader, exitFunc));
      }
    }
    return buildEventTransportsBuilder.build();
  }

  /** Returns true if this factory BuildEventTransport is enabled by the specified options. */
  protected abstract boolean enabled(BuildEventStreamOptions options);

  /** Creates a BuildEventTransport from the specified options. */
  protected abstract BuildEventTransport create(
      BuildEventStreamOptions options,
      BuildEventProtocolOptions protocolOptions,
      BuildEventArtifactUploader uploader,
      Consumer<AbruptExitException> exitFunc)
      throws IOException;

  protected abstract boolean usePathConverter(BuildEventStreamOptions options);
}
