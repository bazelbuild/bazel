/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

import java.nio.ByteBuffer;

/**
 * Transformations on dictionary words.
 */
final class Transform {

  static final int NUM_TRANSFORMS = 121;
  private static final int[] TRANSFORMS = new int[NUM_TRANSFORMS * 3];
  private static final byte[] PREFIX_SUFFIX = new byte[217];
  private static final int[] PREFIX_SUFFIX_HEADS = new int[51];

  // Bundle of 0-terminated strings.
  private static final String PREFIX_SUFFIX_SRC = "# #s #, #e #.# the #.com/#\u00C2\u00A0# of # and"
      + " # in # to #\"#\">#\n#]# for # a # that #. # with #'# from # by #. The # on # as # is #ing"
      + " #\n\t#:#ed #(# at #ly #=\"# of the #. This #,# not #er #al #='#ful #ive #less #est #ize #"
      + "ous #";
  private static final String TRANSFORMS_SRC = "     !! ! ,  *!  &!  \" !  ) *   * -  ! # !  #!*!  "
      + "+  ,$ !  -  %  .  / #   0  1 .  \"   2  3!*   4%  ! # /   5  6  7  8 0  1 &   $   9 +   : "
      + " ;  < '  !=  >  ?! 4  @ 4  2  &   A *# (   B  C& ) %  ) !*# *-% A +! *.  D! %'  & E *6  F "
      + " G% ! *A *%  H! D  I!+!  J!+   K +- *4! A  L!*4  M  N +6  O!*% +.! K *G  P +%(  ! G *D +D "
      + " Q +# *K!*G!+D!+# +G +A +4!+% +K!+4!*D!+K!*K";

  private static void unpackTransforms(byte[] prefixSuffix, int[] prefixSuffixHeads,
      int[] transforms, String prefixSuffixSrc, String transformsSrc) {
    int n = prefixSuffixSrc.length();
    int index = 1;
    for (int i = 0; i < n; ++i) {
      char c = prefixSuffixSrc.charAt(i);
      prefixSuffix[i] = (byte) c;
      if (c == 35) { // == #
        prefixSuffixHeads[index++] = i + 1;
        prefixSuffix[i] = 0;
      }
    }

    for (int i = 0; i < NUM_TRANSFORMS * 3; ++i) {
      transforms[i] = transformsSrc.charAt(i) - 32;
    }
  }

  static {
    unpackTransforms(PREFIX_SUFFIX, PREFIX_SUFFIX_HEADS, TRANSFORMS, PREFIX_SUFFIX_SRC,
        TRANSFORMS_SRC);
  }

  static int transformDictionaryWord(byte[] dst, int dstOffset, ByteBuffer data, int wordOffset,
      int len, int transformIndex) {
    int offset = dstOffset;
    int transformOffset = 3 * transformIndex;
    int transformPrefix = PREFIX_SUFFIX_HEADS[TRANSFORMS[transformOffset]];
    int transformType = TRANSFORMS[transformOffset + 1];
    int transformSuffix = PREFIX_SUFFIX_HEADS[TRANSFORMS[transformOffset + 2]];

    // Copy prefix.
    while (PREFIX_SUFFIX[transformPrefix] != 0) {
      dst[offset++] = PREFIX_SUFFIX[transformPrefix++];
    }

    // Copy trimmed word.
    int omitFirst = transformType >= 12 ? (transformType - 11) : 0;
    if (omitFirst > len) {
      omitFirst = len;
    }
    wordOffset += omitFirst;
    len -= omitFirst;
    len -= transformType <= 9 ? transformType : 0;  // Omit last.
    int i = len;
    while (i > 0) {
      dst[offset++] = data.get(wordOffset++);
      i--;
    }

    // Ferment.
    if (transformType == 11 || transformType == 10) {
      int uppercaseOffset = offset - len;
      if (transformType == 10) {
        len = 1;
      }
      while (len > 0) {
        int tmp = dst[uppercaseOffset] & 0xFF;
        if (tmp < 0xc0) {
          if (tmp >= 97 && tmp <= 122) { // in [a..z] range
            dst[uppercaseOffset] ^= (byte) 32;
          }
          uppercaseOffset += 1;
          len -= 1;
        } else if (tmp < 0xe0) {
          dst[uppercaseOffset + 1] ^= (byte) 32;
          uppercaseOffset += 2;
          len -= 2;
        } else {
          dst[uppercaseOffset + 2] ^= (byte) 5;
          uppercaseOffset += 3;
          len -= 3;
        }
      }
    }

    // Copy suffix.
    while (PREFIX_SUFFIX[transformSuffix] != 0) {
      dst[offset++] = PREFIX_SUFFIX[transformSuffix++];
    }

    return offset - dstOffset;
  }
}
