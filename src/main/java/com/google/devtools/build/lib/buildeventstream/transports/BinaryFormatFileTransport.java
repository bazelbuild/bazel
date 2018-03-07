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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.buildeventstream.ArtifactGroupNamer;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;

/**
 * A simple {@link BuildEventTransport} that writes a varint delimited binary representation of
 * {@link BuildEvent} protocol buffers to a file.
 */
public final class BinaryFormatFileTransport extends FileTransport {

  private final PathConverter pathConverter;

  BinaryFormatFileTransport(String path, PathConverter pathConverter) {
    super(path);
    this.pathConverter = pathConverter;
  }

  @Override
  public String name() {
    return this.getClass().getSimpleName();
  }

  @Override
  public synchronized void sendBuildEvent(BuildEvent event, final ArtifactGroupNamer namer) {
    checkNotNull(event);
    BuildEventConverters converters =
        new BuildEventConverters() {
          @Override
          public PathConverter pathConverter() {
            return pathConverter;
          }
          @Override
          public ArtifactGroupNamer artifactGroupNamer() {
            return namer;
          }
        };
    write(event.asStreamProto(converters));
  }
}
