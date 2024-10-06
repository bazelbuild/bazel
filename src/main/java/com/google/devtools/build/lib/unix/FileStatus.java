// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.unix;

/**
 * Equivalent to UNIX's "struct stat", a FileStatus instance contains various bits of metadata about
 * a directory entry.
 *
 * <p>The Java SDK provides access to some but not all of the information available via the stat(2)
 * and lstat(2) syscalls, but often requires that multiple calls be made to obtain it. By reifying
 * stat buffers as Java objects and providing a wrapper around the stat/lstat calls, we give client
 * applications access to the richer file metadata and enable a reduction in the number of system
 * calls, which is critical for high-performance tools.
 *
 * <p>This class is optimized for memory usage. Fields not required by Bazel are omitted.
 */
public class FileStatus {

  private final int mode;
  private final long atime; // milliseconds since Unix epoch
  private final long mtime; // milliseconds since Unix epoch
  private final long ctime; // milliseconds since Unix epoch
  private final long size;
  private final int dev;
  private final long ino;

  /** Constructs a FileStatus instance. (Called only from ErrnoFileStatus and JNI code.) */
  protected FileStatus(int mode, long atime, long mtime, long ctime, long size, int dev, long ino) {
    this.mode = mode;
    this.atime = atime;
    this.mtime = mtime;
    this.ctime = ctime;
    this.size = size;
    this.dev = dev;
    this.ino = ino;
  }

  /**
   * Returns the device number of this inode.
   */
  public int getDeviceNumber() {
    return dev;
  }

  /**
   * Returns the number of this inode.  Inode numbers are (usually) unique for
   * a given device.
   */
  public long getInodeNumber() {
    return ino;
  }

  /**
   * Returns true iff this file is a regular file.
   */
  public boolean isRegularFile() {
    return (mode & S_IFMT) == S_IFREG;
  }

  /** Returns true iff this file is a directory. */
  public boolean isDirectory() {
    return (mode & S_IFMT) == S_IFDIR;
  }

  public static boolean isDirectory(int rawType) {
    int type = rawType & S_IFMT;
    return type == S_IFDIR;
  }

  /** Returns true iff this file is a symbolic link. */
  public boolean isSymbolicLink() {
    return (mode & S_IFMT) == S_IFLNK;
  }

  public static boolean isSymbolicLink(int rawType) {
    int type = rawType & S_IFMT;
    return type == S_IFLNK;
  }

  /**
   * Returns true iff this file is a character device.
   */
  public boolean isCharacterDevice() {
    return (mode & S_IFMT) == S_IFCHR;
  }

  /**
   * Returns true iff this file is a block device.
   */
  public boolean isBlockDevice() {
    return (mode & S_IFMT) == S_IFBLK;
  }

  /** Returns true iff this file is a FIFO. */
  public boolean isFifo() {
    return (mode & S_IFMT) == S_IFIFO;
  }

  /**
   * Returns true iff this file is a UNIX-domain socket.
   */
  public boolean isSocket() {
    return (mode & S_IFMT) == S_IFSOCK;
  }

  /**
   * Returns true iff this file has its "set UID" bit set.
   */
  public boolean isSetUserId() {
    return (mode & S_ISUID) != 0;
  }

  /**
   * Returns true iff this file has its "set GID" bit set.
   */
  public boolean isSetGroupId() {
    return (mode & S_ISGID) != 0;
  }

  /**
   * Returns true iff this file has its "sticky" bit set.  See UNIX manuals for
   * explanation.
   */
  public boolean isSticky() {
    return (mode & S_ISVTX) != 0;
  }

  /**
   * Returns the user/group/other permissions part of the mode bits (i.e. mode masked with 0777),
   * interpreted according to longstanding UNIX tradition.
   */
  public int getPermissions() {
    return mode & S_IRWXA;
  }

  /**
   * Returns the total size, in bytes, of this file.
   */
  public long getSize() {
    return size;
  }

  /** Returns the last access time of this file (milliseconds since UNIX epoch). */
  public long getLastAccessTime() {
    return atime;
  }

  /** Returns the last modified time of this file (milliseconds since UNIX epoch). */
  public long getLastModifiedTime() {
    return mtime;
  }

  /** Returns the last change time of this file (milliseconds since UNIX epoch). */
  public long getLastChangeTime() {
    return ctime;
  }

  ////////////////////////////////////////////////////////////////////////

  @Override
  public String toString() {
    return String.format(
        "FileStatus(mode=0%06o,atime=%d,mtime=%d,ctime=%d,size=%d,device=%d,ino=%d)",
        mode, atime, mtime, ctime, size, dev, ino);
  }

  ////////////////////////////////////////////////////////////////////////
  // Platform-specific details. These fields are public so that they can
  // be used from other packages. See POSIX and/or Linux manuals for details.
  //
  // These need to be kept in sync with the native code and system call
  // interface.  (The unit tests ensure that.)  Of course, this decoding could
  // be done in the JNI code to ensure maximum portability, but (a) we don't
  // expect we'll need that any time soon, and (b) that would require eager
  // rather than on-demand bitmunging of all attributes.  In any case, it's not
  // part of the interface so it can be easily changed later if necessary.

  public static final int S_IFMT =   0170000; // mask: filetype bitfields
  public static final int S_IFSOCK = 0140000; // socket
  public static final int S_IFLNK =  0120000; // symbolic link
  public static final int S_IFREG =  0100000; // regular file
  public static final int S_IFBLK =  0060000; // block device
  public static final int S_IFDIR =  0040000; // directory
  public static final int S_IFCHR =  0020000; // character device
  public static final int S_IFIFO =  0010000; // fifo
  public static final int S_ISUID =  0004000; // set UID bit
  public static final int S_ISGID =  0002000; // set GID bit (see below)
  public static final int S_ISVTX =  0001000; // sticky bit (see below)
  public static final int S_IRWXA =  00777; // mask: all permissions
  public static final int S_IRWXU =  00700; // mask: file owner permissions
  public static final int S_IRUSR =  00400; // owner has read permission
  public static final int S_IWUSR =  00200; // owner has write permission
  public static final int S_IXUSR =  00100; // owner has execute permission
  public static final int S_IRWXG =  00070; // mask: group permissions
  public static final int S_IRGRP =  00040; // group has read permission
  public static final int S_IWGRP =  00020; // group has write permission
  public static final int S_IXGRP =  00010; // group has execute permission
  public static final int S_IRWXO =  00007; // mask: other permissions
  public static final int S_IROTH =  00004; // others have read permission
  public static final int S_IWOTH =  00002; // others have write permisson
  public static final int S_IXOTH =  00001; // others have execute permission

  public static final int S_IEXEC =  00111; // owner, group, world execute

  public static boolean isFile(int rawType) {
    int type = rawType & S_IFMT;
    return type == S_IFREG || isSpecialFile(rawType);
  }

  public static boolean isSpecialFile(int rawType) {
    int type = rawType & S_IFMT;
    return type == S_IFSOCK || type == S_IFBLK || type == S_IFCHR || type == S_IFIFO;
  }
}
