// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.errorprone.annotations.Immutable;
import com.google.gson.JsonSyntaxException;
import com.google.gson.TypeAdapter;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.net.URI;
import java.util.Locale;

/**
 * Request for the {@code get} command of the <a
 * href="https://github.com/bazelbuild/proposals/blob/main/designs/2022-06-07-bazel-credential-helpers.md#proposal">Credential
 * Helper Protocol</a>.
 */
@AutoValue
@AutoValue.CopyAnnotations
@Immutable
@JsonAdapter(GetCredentialsRequest.GsonTypeAdapter.class)
public abstract class GetCredentialsRequest {
  /** Returns the {@link URI} this request is for. */
  public abstract URI getUri();

  /** Returns a new builder for {@link GetCredentialsRequest}. */
  public static Builder newBuilder() {
    return new AutoValue_GetCredentialsRequest.Builder();
  }

  /** Builder for {@link GetCredentialsRequest}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the {@link URI} this request is for. */
    public abstract Builder setUri(URI uri);

    /** Returns the newly constructed {@link GetCredentialsRequest}. */
    public abstract GetCredentialsRequest build();
  }

  /** GSON adapter for GetCredentialsRequest. */
  public static final class GsonTypeAdapter extends TypeAdapter<GetCredentialsRequest> {
    @Override
    public void write(JsonWriter writer, GetCredentialsRequest value) throws IOException {
      Preconditions.checkNotNull(writer);
      Preconditions.checkNotNull(value);

      writer.beginObject();
      writer.name("uri").value(value.getUri().toString());
      writer.endObject();
    }

    @Override
    public GetCredentialsRequest read(JsonReader reader) throws IOException {
      Preconditions.checkNotNull(reader);

      Builder request = newBuilder();

      if (reader.peek() != JsonToken.BEGIN_OBJECT) {
        throw new JsonSyntaxException(
            String.format(Locale.US, "Expected object, got %s", reader.peek()));
      }
      reader.beginObject();
      while (reader.hasNext()) {
        String name = reader.nextName();
        switch (name) {
          case "uri" -> {
            if (reader.peek() != JsonToken.STRING) {
              throw new JsonSyntaxException(
                  String.format(
                      Locale.US, "Expected value of 'url' to be a string, got %s", reader.peek()));
            }
            request.setUri(URI.create(reader.nextString()));
          }
          default ->
              // We intentionally ignore unknown keys to achieve forward compatibility with requests
              // coming from newer tools.
              reader.skipValue();
        }
      }
      reader.endObject();
      return request.build();
    }
  }
}
