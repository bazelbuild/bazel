// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.devtools.build.lib.util.Preconditions;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;

/**
 * Abstract base class for {@link PathFragment} instances that will be allocated when Blaze is run
 * on a non-Windows platform.
 */
abstract class UnixPathFragment extends PathFragment {
  static final Helper HELPER = new Helper();
  /**
    * We have two concrete subclasses with zero per-instance additional memory overhead. Do not add
    * any fields. See the comment on memory use in PathFragment for details on the current
    * per-instance memory usage.
    */

  protected UnixPathFragment(String[] segments) {
    super(segments);
  }

  @Override
  protected int computeHashCode() {
    int h = 0;
    for (String segment : segments) {
      int segmentHash = segment.hashCode();
      h = h * 31 + segmentHash;
    }
    return h;
  }

  @Override
  public String windowsVolume() {
    return "";
  }

  @Override
  public char getDriveLetter() {
    return '\0';
  }

  private static class Helper extends PathFragment.Helper {
    private static final char SEPARATOR_CHAR = '/';

    @Override
    PathFragment create(String path) {
      boolean isAbsolute = path.length() > 0 && isSeparator(path.charAt(0));
      return isAbsolute
          ? new AbsoluteUnixPathFragment(segment(path, 1))
          : new RelativeUnixPathFragment(segment(path, 0));
    }

    @Override
    PathFragment createAlreadyInterned(char driveLetter, boolean isAbsolute, String[] segments) {
      Preconditions.checkState(driveLetter == '\0', driveLetter);
      return isAbsolute
          ? new AbsoluteUnixPathFragment(segments)
          : new RelativeUnixPathFragment(segments);
    }

    @Override
    char getPrimarySeparatorChar() {
      return SEPARATOR_CHAR;
    }

    @Override
    boolean isSeparator(char c) {
      return c == SEPARATOR_CHAR;
    }

    @Override
    boolean containsSeparatorChar(String path) {
      return path.indexOf(SEPARATOR_CHAR) != -1;
    }

    @Override
    boolean segmentsEqual(int length, String[] segments1, int offset1, String[] segments2) {
      if ((segments1.length - offset1) < length || segments2.length < length) {
        return false;
      }
      for (int i = 0; i < length; ++i) {
        String seg1 = segments1[i + offset1];
        String seg2 = segments2[i];
        if ((seg1 == null) != (seg2 == null)) {
          return false;
        }
        if (seg1 == null) {
          continue;
        }
        if (!seg1.equals(seg2)) {
          return false;
        }
      }
      return true;
    }

    @Override
    protected int compare(PathFragment pathFragment1, PathFragment pathFragment2) {
      if (pathFragment1.isAbsolute() != pathFragment2.isAbsolute()) {
        return pathFragment1.isAbsolute() ? -1 : 1;
      }
      String[] segments1 = pathFragment1.segments();
      String[] segments2 = pathFragment2.segments();
      int len1 = segments1.length;
      int len2 = segments2.length;
      int n = Math.min(len1, len2);
      for (int i = 0; i < n; i++) {
        String seg1 = segments1[i];
        String seg2 = segments2[i];
        int cmp = seg1.compareTo(seg2);
        if (cmp != 0) {
          return cmp;
        }
      }
      return len1 - len2;
    }
  }

  private static final class AbsoluteUnixPathFragment extends UnixPathFragment {
    private AbsoluteUnixPathFragment(String[] segments) {
      super(segments);
    }

    @Override
    public boolean isAbsolute() {
      return true;
    }

    @Override
    protected int computeHashCode() {
      int h = Boolean.TRUE.hashCode();
      h = h * 31 + super.computeHashCode();
      return h;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof AbsoluteUnixPathFragment)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      AbsoluteUnixPathFragment otherAbsoluteUnixPathFragment = (AbsoluteUnixPathFragment) other;
      return HELPER.segmentsEqual(this.segments, otherAbsoluteUnixPathFragment.segments);
    }

    // Java serialization looks for the presence of this method in the concrete class. It is not
    // inherited from the parent class.
    @Override
    protected Object writeReplace() {
      return super.writeReplace();
    }

    // Java serialization looks for the presence of this method in the concrete class. It is not
    // inherited from the parent class.
    @Override
    protected void readObject(ObjectInputStream stream) throws InvalidObjectException {
      super.readObject(stream);
    }
  }

  private static final class RelativeUnixPathFragment extends UnixPathFragment {
    private RelativeUnixPathFragment(String[] segments) {
      super(segments);
    }

    @Override
    public boolean isAbsolute() {
      return false;
    }

    @Override
    protected int computeHashCode() {
      int h = Boolean.FALSE.hashCode();
      h = h * 31 + super.computeHashCode();
      return h;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof RelativeUnixPathFragment)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      RelativeUnixPathFragment otherRelativeUnixPathFragment = (RelativeUnixPathFragment) other;
      return HELPER.segmentsEqual(this.segments, otherRelativeUnixPathFragment.segments);
    }

    // Java serialization looks for the presence of this method in the concrete class. It is not
    // inherited from the parent class.
    @Override
    protected Object writeReplace() {
      return super.writeReplace();
    }

    // Java serialization looks for the presence of this method in the concrete class. It is not
    // inherited from the parent class.
    @Override
    protected void readObject(ObjectInputStream stream) throws InvalidObjectException {
      super.readObject(stream);
    }
  }
}
