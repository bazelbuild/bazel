package com.google.devtools.build.lib.bazel.bzlmod.jsontypefactories;

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Registry;
import com.google.devtools.build.lib.bazel.bzlmod.RegistryFactory;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

public class TypeAdapterUtil {

  public static TypeAdapter<Version> versionTypeAdapter = new TypeAdapter<>() {
    @Override
    public void write(JsonWriter jsonWriter, Version version) throws IOException {
      jsonWriter.value(version.toString());
    }

    @Override
    public Version read(JsonReader jsonReader) throws IOException {
      Version version;
      String versionString = jsonReader.nextString();
      try {
        version = Version.parse(versionString);
      } catch (ParseException e) {
        throw new JsonParseException(String.format("Unable to parse Version %s from the lockfile", versionString), e);
      }
      return version;
    }
  };

  public static TypeAdapter<ModuleKey> moduleKeyTypeAdapter = new TypeAdapter<>() {
    @Override
    public void write(JsonWriter jsonWriter, ModuleKey moduleKey) throws IOException {
      jsonWriter.value(moduleKey.toString());
    }

    @Override
    public ModuleKey read(JsonReader jsonReader) throws IOException {
      String jsonString = jsonReader.nextString();
      if (jsonString.equals("<root>")) {
        return ModuleKey.ROOT;
      }
      List<String> parts = Splitter.on('@').splitToList(jsonString);
      if(parts.get(1).equals("_")) {
        return ModuleKey.create(parts.get(0), Version.EMPTY);
      }

      Version version;
      try {
        version = Version.parse(parts.get(1));
      } catch (ParseException e) {
        throw new JsonParseException(String.format("Unable to parse ModuleKey %s version from the lockfile", jsonString), e);
      }
      return ModuleKey.create(parts.get(0), version);
    }
  };

  public static TypeAdapter<Registry> registryTypeAdapter(RegistryFactory registryFactory) {
    return new TypeAdapter<>() {
      @Override
      public void write(JsonWriter jsonWriter, Registry registry) throws IOException {
        jsonWriter.value(registry.getUrl());
      }

      @Override
      public Registry read(JsonReader jsonReader) throws IOException {
        try {
          return registryFactory.getRegistryWithUrl(jsonReader.nextString());
        } catch (URISyntaxException e) {
          throw new RuntimeException("Lockfile registry URL is not valid", e);
        }
      }
    };
  }

  private static GsonBuilder adapterGson = new GsonBuilder()
      .registerTypeAdapterFactory(GenerateTypeAdapter.FACTORY)
      .registerTypeAdapterFactory(new DictTypeAdapterFactory())
      .registerTypeAdapterFactory(new ImmutableMapTypeAdapterFactory())
      .registerTypeAdapterFactory(new ImmutableListTypeAdatperFactory())
      .registerTypeAdapterFactory(new ImmutableBiMapTypeAdapterFactory())
      .registerTypeAdapter(Version.class, versionTypeAdapter)
      .registerTypeAdapter(ModuleKey.class, moduleKeyTypeAdapter);

  public static Gson getLockfileGsonWithTypeAdapters(RegistryFactory registryFactory){
    return adapterGson
        .registerTypeAdapter(Registry.class, registryTypeAdapter(registryFactory))
        .create();
  }

}
