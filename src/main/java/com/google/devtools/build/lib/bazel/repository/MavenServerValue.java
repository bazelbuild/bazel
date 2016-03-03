// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Objects;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import org.apache.maven.settings.Server;

import java.util.Arrays;

/**
 * A Maven repository's identifier.
 */
public class MavenServerValue implements SkyValue {
  public static final String DEFAULT_ID = "default";

  private final String id;
  private final String url;
  private final Server server;
  private final byte[] settingsFingerprint;

  public static SkyKey key(String serverName) {
    Preconditions.checkNotNull(serverName);
    return SkyKey.create(MavenServerFunction.NAME, serverName);
  }

  public static MavenServerValue createFromUrl(String url) {
    return new MavenServerValue(DEFAULT_ID, url, new Server(),
        new Fingerprint().digestAndReset());
  }

  public MavenServerValue(String id, String url, Server server, byte[] settingsFingerprint) {
    Preconditions.checkNotNull(id);
    Preconditions.checkNotNull(url);
    Preconditions.checkNotNull(server);
    this.id = id;
    this.url = url;
    this.server = server;
    this.settingsFingerprint = settingsFingerprint;
  }

  @Override
  public boolean equals(Object object) {
    if (this == object) {
      return true;
    }
    if (object == null || !(object instanceof MavenServerValue)) {
      return false;
    }

    MavenServerValue other = (MavenServerValue) object;
    return id.equals(other.id) && url.equals(other.url)
        && Arrays.equals(settingsFingerprint, other.settingsFingerprint);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode(id, url);
  }

  public String getUrl() {
    return url;
  }

  public Server getServer() {
    return server;
  }

  public byte[] getSettingsFingerprint() {
    return settingsFingerprint;
  }
}
