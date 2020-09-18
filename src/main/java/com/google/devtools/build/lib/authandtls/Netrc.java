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
package com.google.devtools.build.lib.authandtls;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

public class Netrc {

  @Nullable
  private final Credential defaultCredential;
  private final ImmutableMap<String, Credential> credentials;

  public static Netrc fromStream(InputStream inputStream) throws IOException  {
    return NetrcParser.parseAndClose(inputStream);
  }

  public Netrc(@Nullable Credential defaultCredential, ImmutableMap<String, Credential> credentials) {
    this.defaultCredential = defaultCredential;
    this.credentials = credentials;
  }

  @Nullable
  public Credential getCredential(String machine) {
    return credentials.getOrDefault(machine, defaultCredential);
  }

  @VisibleForTesting
  @Nullable
  Credential getDefaultCredential() {
    return defaultCredential;
  }

  @VisibleForTesting
  ImmutableMap<String, Credential> getCredentials() {
    return credentials;
  }

  @AutoValue
  public static abstract class Credential {

    public static Builder builder(String machine) {
      return new AutoValue_Netrc_Credential.Builder()
          .setMachine(machine)
          .setLogin("")
          .setPassword("")
          .setAccount("");
    }

    @AutoValue.Builder
    public static abstract class Builder {
      public abstract String machine();
      public abstract Builder setMachine(String machine);

      public abstract String login();
      public abstract Builder setLogin(String value);

      public abstract String password();
      public abstract Builder setPassword(String value);

      public abstract String account();
      public abstract Builder setAccount(String value);

      public abstract Credential build();
    }

    abstract String machine();
    abstract String login();
    abstract String password();
    abstract String account();
  }
}
