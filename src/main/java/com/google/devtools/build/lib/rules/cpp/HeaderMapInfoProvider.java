package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.util.FileTypeSet;

public class HeaderMapInfoProvider implements TransitiveInfoProvider {
  private HeaderMapInfo info;

  public static HeaderMapInfoProvider EMPTY = new HeaderMapInfoProvider(HeaderMapInfo.EMPTY);

  public HeaderMapInfo getInfo() {
    return info;
  }

  private HeaderMapInfoProvider(HeaderMapInfo info) {
    this.info = info;
  }

  /** Builder for HeaderMapInfoProvider */
  public static class Builder {
    private HeaderMapInfo info;

    public Builder setHeaderMapInfo(HeaderMapInfo info) {
      this.info = info;
      return this;
    }

    public HeaderMapInfoProvider build() {
      return new HeaderMapInfoProvider(this.info);
    }
  }
}
