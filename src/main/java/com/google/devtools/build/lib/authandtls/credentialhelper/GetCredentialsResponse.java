package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.errorprone.annotations.Immutable;
import com.google.gson.Gson;
import com.google.gson.TypeAdapter;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Response from the {@code get} command of the <a
 * href="https://github.com/bazelbuild/proposals/blob/main/designs/2022-06-07-bazel-credential-helpers.md#proposal">Credential
 * Helper Protocol</a>.
 */
@AutoValue
@AutoValue.CopyAnnotations
@Immutable
@JsonAdapter(GetCredentialsResponse.GsonTypeAdapter.class)
public abstract class GetCredentialsResponse {
  /** Returns the headers to attach to the request. */
  public abstract ImmutableMap<String, ImmutableList<String>> getHeaders();

  /** Returns a new builder for {@link GetCredentialsRequest}. */
  public static Builder newBuilder() {
    return new AutoValue_GetCredentialsResponse.Builder();
  }

  /** Builder for {@link GetCredentialsResponse}. */
  @AutoValue.Builder
  public abstract static class Builder {
    protected abstract ImmutableMap.Builder<String, ImmutableList<String>> headersBuilder();

    /** Returns the newly constructed {@link GetCredentialsResponse}. */
    public abstract GetCredentialsResponse build();
  }

  public static final class GsonTypeAdapter extends TypeAdapter<GetCredentialsResponse> {
    private static final Gson GSON = new Gson();

    @Override
    public void write(JsonWriter writer, GetCredentialsResponse response) throws IOException {
      Preconditions.checkNotNull(writer);
      Preconditions.checkNotNull(response);

      writer.beginObject();

      ImmutableMap<String, ImmutableList<String>> headers = response.getHeaders();
      if (!headers.isEmpty()) {
        writer.name("headers");
        writer.beginObject();
        for (Map.Entry<String, ImmutableList<String>> entry : headers.entrySet()) {
          writer.name(entry.getKey());

          writer.beginArray();
          for (String value : entry.getValue()) {
            writer.value(value);
          }
          writer.endArray();
        }
        writer.endObject();
      }
      writer.endObject();
    }

    @Override
    public GetCredentialsResponse read(JsonReader reader) throws IOException {
      Preconditions.checkNotNull(reader);

      GetCredentialsResponse.Builder response = newBuilder();
      reader.beginObject();
      while (reader.hasNext()) {
        String name = reader.nextName();
        switch (name) {
          case "headers":
            for (Map.Entry<String, List<String>> header : readHeaders(reader).entrySet()) {
              response
                  .headersBuilder()
                  .put(header.getKey(), ImmutableList.copyOf(header.getValue()));
            }
            break;

          default:
            // We intentionally ignore unknown keys to achieve forward compatibility with responses
            // coming from newer tools.
            reader.skipValue();
        }
      }
      reader.endObject();
      return response.build();
    }

    private Map<String, List<String>> readHeaders(JsonReader reader) throws IOException {
      Preconditions.checkNotNull(reader);

      return GSON.getAdapter(Map.class).read(reader);
    }
  }
}
