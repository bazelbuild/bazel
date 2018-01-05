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

import java.io.InvalidObjectException;
import java.io.ObjectInputStream;

/**
 * Abstract base class for {@link PathFragment} instances that will be allocated when Blaze is run
 * on a Windows platform.
 */
abstract class WindowsPathFragment extends PathFragment {
  static final Helper HELPER = new Helper();

  // The drive letter of an absolute path, eg. 'C' for 'C:/foo'.
  // We deliberately ignore "C:foo" style paths and treat them like a literal "C:foo" path segment.
  protected final char driveLetter;

  protected WindowsPathFragment(char driveLetter, String[] segments) {
    super(segments);
    this.driveLetter = driveLetter;
  }

  @Override
  public String windowsVolume() {
    return (driveLetter != '\0') ? driveLetter + ":" : "";
  }

  @Override
  public char getDriveLetter() {
    return driveLetter;
  }

  @Override
  protected int computeHashCode() {
    int h = 0;
    for (String segment : segments) {
      int segmentHash = segment.toLowerCase().hashCode();
      h = h * 31 + segmentHash;
    }
    return h;
  }

  private static class Helper extends PathFragment.Helper {
    private static final char SEPARATOR_CHAR = '/';
    // TODO(laszlocsomor): Lots of internal PathFragment operations, e.g. getPathString, use the
    // primary separator char and do not use this.
    private static final char EXTRA_SEPARATOR_CHAR = '\\';

    private static boolean isDriveLetter(char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }

    @Override
    PathFragment create(String path) {
      char driveLetter =
          path.length() >= 3
                  && path.charAt(1) == ':'
                  && isSeparator(path.charAt(2))
                  && isDriveLetter(path.charAt(0))
              ? Character.toUpperCase(path.charAt(0))
              : '\0';
      if (driveLetter != '\0') {
        path = path.substring(2);
      }
      boolean isAbsolute = path.length() > 0 && isSeparator(path.charAt(0));
      return isAbsolute
          ? new AbsoluteWindowsPathFragment(driveLetter, segment(path, 1))
          : new RelativeWindowsPathFragment(driveLetter, segment(path, 0));
    }

    @Override
    PathFragment createAlreadyInterned(char driveLetter, boolean isAbsolute, String[] segments) {
      return isAbsolute
          ? new AbsoluteWindowsPathFragment(driveLetter, segments)
          : new RelativeWindowsPathFragment(driveLetter, segments);
    }

    @Override
    char getPrimarySeparatorChar() {
      return SEPARATOR_CHAR;
    }

    @Override
    boolean isSeparator(char c) {
      return c == SEPARATOR_CHAR || c == EXTRA_SEPARATOR_CHAR;
    }

    @Override
    boolean containsSeparatorChar(String path) {
      // TODO(laszlocsomor): This is inefficient.
      return path.indexOf(SEPARATOR_CHAR) != -1 || path.indexOf(EXTRA_SEPARATOR_CHAR) != -1;
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
        // TODO(laszlocsomor): The calls to String#toLowerCase are inefficient and potentially
        // repeated too. Also, why not use String#equalsIgnoreCase.
        seg1 = seg1.toLowerCase();
        seg2 = seg2.toLowerCase();
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
      int cmp = Character.compare(pathFragment1.getDriveLetter(), pathFragment2.getDriveLetter());
      if (cmp != 0) {
        return cmp;
      }
      String[] segments1 = pathFragment1.segments();
      String[] segments2 = pathFragment2.segments();
      int len1 = segments1.length;
      int len2 = segments2.length;
      int n = Math.min(len1, len2);
      for (int i = 0; i < n; i++) {
        String seg1 = segments1[i].toLowerCase();
        String seg2 = segments2[i].toLowerCase();
        cmp = seg1.compareTo(seg2);
        if (cmp != 0) {
          return cmp;
        }
      }
      return len1 - len2;
    }
  }

  private static final class AbsoluteWindowsPathFragment extends WindowsPathFragment {
    private AbsoluteWindowsPathFragment(char driveLetter, String[] segments) {
      super(driveLetter, segments);
    }

    @Override
    public boolean isAbsolute() {
      return true;
    }

    @Override
    protected int computeHashCode() {
      int h = Boolean.TRUE.hashCode();
      h = h * 31 + super.computeHashCode();
      h = h * 31 + Character.valueOf(getDriveLetter()).hashCode();
      return h;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof AbsoluteWindowsPathFragment)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      AbsoluteWindowsPathFragment otherAbsoluteWindowsPathFragment =
          (AbsoluteWindowsPathFragment) other;
      return this.driveLetter == otherAbsoluteWindowsPathFragment.driveLetter
          && HELPER.segmentsEqual(this.segments, otherAbsoluteWindowsPathFragment.segments);
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

  private static final class RelativeWindowsPathFragment extends WindowsPathFragment {
    private RelativeWindowsPathFragment(char driveLetter, String[] segments) {
      super(driveLetter, segments);
    }

    @Override
    public boolean isAbsolute() {
      return false;
    }

    @Override
    protected int computeHashCode() {
      int h = Boolean.FALSE.hashCode();
      h = h * 31 + super.computeHashCode();
      if (!isEmpty()) {
        h = h * 31 + Character.valueOf(getDriveLetter()).hashCode();
      }
      return h;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof RelativeWindowsPathFragment)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      RelativeWindowsPathFragment otherRelativeWindowsPathFragment =
          (RelativeWindowsPathFragment) other;
      return isEmpty() && otherRelativeWindowsPathFragment.isEmpty()
          ? true
          : this.driveLetter == otherRelativeWindowsPathFragment.driveLetter
              && HELPER.segmentsEqual(this.segments, otherRelativeWindowsPathFragment.segments);
    }

    private boolean isEmpty() {
      return segmentCount() == 0;
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
