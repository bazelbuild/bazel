# Copyright (c) 2015, Google Inc.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import os.path
import shutil
import sys
import tarfile
import tempfile
import urllib

# CLANG_REVISION and CLANG_SUB_REVISION determine the build of clang
# to use. These should be synced with tools/clang/scripts/update.sh in
# Chromium.
CLANG_REVISION = "233105"
CLANG_SUB_REVISION = "1"

PACKAGE_VERSION = "%s-%s" % (CLANG_REVISION, CLANG_SUB_REVISION)
LLVM_BUILD_DIR = os.path.join(os.path.dirname(__file__), "llvm-build")
STAMP_FILE = os.path.join(LLVM_BUILD_DIR, "cr_build_revision")

CDS_URL = "https://commondatastorage.googleapis.com/chromium-browser-clang"

def DownloadFile(url, path):
  """DownloadFile fetches |url| to |path|."""
  last_progress = [0]
  def report(a, b, c):
    progress = int(a * b * 100.0 / c)
    if progress != last_progress[0]:
      print >> sys.stderr, "Downloading... %d%%" % progress
      last_progress[0] = progress
  urllib.urlretrieve(url, path, reporthook=report)

def main(args):
  # For now, only download clang on Linux.
  if not sys.platform.startswith("linux"):
    return 0

  if os.path.exists(STAMP_FILE):
    with open(STAMP_FILE) as f:
      if f.read().strip() == PACKAGE_VERSION:
        print >> sys.stderr, "Clang already at %s" % (PACKAGE_VERSION,)
        return 0

  if os.path.exists(LLVM_BUILD_DIR):
    shutil.rmtree(LLVM_BUILD_DIR)

  print >> sys.stderr, "Downloading Clang %s" % (PACKAGE_VERSION,)
  cds_full_url = "%s/Linux_x64/clang-%s.tgz" % (CDS_URL, PACKAGE_VERSION)
  with tempfile.NamedTemporaryFile() as temp:
    DownloadFile(cds_full_url, temp.name)
    with tarfile.open(temp.name, "r:gz") as tar_file:
      tar_file.extractall(LLVM_BUILD_DIR)

  with open(STAMP_FILE, "wb") as stamp_file:
    stamp_file.write(PACKAGE_VERSION)

  return 0

if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
