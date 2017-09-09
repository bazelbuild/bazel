package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableList;

public class HeaderMapInfo {
  private final ImmutableMap<String, String> sources;

  public final static HeaderMapInfo EMPTY = new HeaderMapInfo(ImmutableMap.of());

  private HeaderMapInfo(ImmutableMap<String, String> sources) {
    this.sources = sources;
  }

  public ImmutableMap<String, String> getSources() {
    return sources;
  }

  /** Builder for HeaderMapInfo */
  public static class Builder {
    private final ImmutableList.Builder<Artifact> basicHeaders = ImmutableList.builder();
    private final ImmutableList.Builder<Artifact> namespacedHeaders = ImmutableList.builder();
    private final ImmutableList.Builder<HeaderMapInfo> mergedHeaderMapInfos = ImmutableList.builder();
    private String namespace = "";

    /** Set the namespace. */
    public Builder setNamespace(String namespace) {
      this.namespace = namespace;
      return this;
    }

    /** Signals that the build uses headers. */
    public Builder addHeaders(Iterable<Artifact> headers) {
      this.basicHeaders.addAll(headers);
      return this;
    }

    /** Signals that the build uses headers under the namespace. */
    public Builder addNamespacedHeaders(Iterable<Artifact> headers) {
      this.namespacedHeaders.addAll(headers);
      return this;
    }

    /**
     * Merge a header map info.
     * Merged HeaderMapInfos are merged in reverse that they were added.
     * Directly added headers take precedence over those that were merged.
     */
    public Builder mergeHeaderMapInfo(HeaderMapInfo info) {
      this.mergedHeaderMapInfos.add(info);
      return this;
    }

    public HeaderMapInfo build() {
      Map inputMap = new HashMap();
      for (HeaderMapInfo info: mergedHeaderMapInfos.build().reverse()){
        for (Map.Entry<String, String> entry: info.getSources().entrySet()){
          inputMap.put(entry.getKey(), entry.getValue());
        }
      }

      for (Artifact hdr : basicHeaders.build()) {
        inputMap.put(hdr.getPath().getBaseName(), hdr.getExecPath().getPathString());
      }

      // If there is no namespace, don't add a slash
      if (namespace.equals("") == false) {
        for (Artifact hdr : namespacedHeaders.build()) {
          String namespacedKey = namespace + "/" + hdr.getPath().getBaseName();
          inputMap.put(namespacedKey, hdr.getExecPath().getPathString());
        }
      } else {
        for (Artifact hdr : namespacedHeaders.build()) {
          inputMap.put(hdr.getPath().getBaseName(), hdr.getExecPath().getPathString());
        }
      }
      return new HeaderMapInfo(ImmutableMap.copyOf(inputMap));
    }
  }
}
