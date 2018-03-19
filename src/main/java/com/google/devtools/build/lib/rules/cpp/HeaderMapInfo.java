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
    private final ImmutableList.Builder<Artifact> includePrefixdHeaders = ImmutableList.builder();
    private final ImmutableList.Builder<HeaderMapInfo> mergedHeaderMapInfos = ImmutableList.builder();
    private String includePrefix = "";

    /** Set include prefix. */
    public Builder setIncludePrefix(String includePrefix) {
      this.includePrefix = includePrefix;
      return this;
    }

    /**
     * Signals that the build uses headers.
     *
     * This is used when `flatten_virtual_headers` is set: these headers will be
     * mapped to "Header.h" -> path/to/Header.
     */
    public Builder addHeaders(Iterable<Artifact> headers) {
      this.basicHeaders.addAll(headers);
      return this;
    }

    /** Signals that the build uses headers under the includePrefix. */
    public Builder addIncludePrefixdHeaders(Iterable<Artifact> headers) {
      this.includePrefixdHeaders.addAll(headers);
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

      // If there is no includePrefix, don't add a slash
      if (includePrefix.equals("") == false) {
        for (Artifact hdr : includePrefixdHeaders.build()) {
          // Set the include prefix:
          // IncludePrefix/Header.h -> Path/To/Header.h
          String includePrefixdKey = includePrefix + "/" + hdr.getPath().getBaseName();
          inputMap.put(includePrefixdKey, hdr.getExecPath().getPathString());
        }
      }
      return new HeaderMapInfo(ImmutableMap.copyOf(inputMap));
    }
  }
}
