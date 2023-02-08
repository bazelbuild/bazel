package com.google.devtools.build.lib.bazel.bzlmod.jsontypefactories;

import com.google.common.base.Splitter;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleKey;
import com.google.devtools.build.lib.bazel.bzlmod.Registry;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.TypeAdapterFactory;
import com.google.gson.reflect.TypeToken;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.io.IOException;
import java.util.List;

public class TypeAdapterUtil {

  public static class RegistryAdapterFactory implements TypeAdapterFactory {
    private final Class<? extends Registry> implementationClass;

    public RegistryAdapterFactory(Class<? extends Registry> implementationClass) {
      this.implementationClass = implementationClass;
    }

    @Override
    public <T> TypeAdapter<T> create(Gson gson, TypeToken<T> type) {
      if (!Registry.class.equals(type.getRawType())) return null;
      return (TypeAdapter<T>) gson.getAdapter(implementationClass);
    }
  }

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

  public static Gson getLockfileGsonWithTypeAdapters(final Class<? extends Registry> registryTpe){
    return new GsonBuilder()
        .registerTypeAdapterFactory(GenerateTypeAdapter.FACTORY)
        .registerTypeAdapterFactory(new DictTypeAdapterFactory())
        .registerTypeAdapterFactory(new ImmutableMapTypeAdapterFactory())
        .registerTypeAdapterFactory(new ImmutableListTypeAdatperFactory())
        .registerTypeAdapterFactory(new ImmutableBiMapTypeAdapterFactory())
        .registerTypeAdapterFactory(new RegistryAdapterFactory(registryTpe))
        .registerTypeAdapter(Version.class, versionTypeAdapter)
        .registerTypeAdapter(ModuleKey.class, moduleKeyTypeAdapter)
        .create();
  }

}
