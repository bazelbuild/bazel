/* Copyright 2015 Google Inc. All Rights Reserved.

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

package org.brotli.dec;

import java.io.IOException;
import java.io.InputStream;

/**
 * API for Brotli decompression.
 */
final class Decode {

  //----------------------------------------------------------------------------
  // RunningState
  //----------------------------------------------------------------------------
  private static final int UNINITIALIZED = 0;
  private static final int BLOCK_START = 1;
  private static final int COMPRESSED_BLOCK_START = 2;
  private static final int MAIN_LOOP = 3;
  private static final int READ_METADATA = 4;
  private static final int COPY_UNCOMPRESSED = 5;
  private static final int INSERT_LOOP = 6;
  private static final int COPY_LOOP = 7;
  private static final int TRANSFORM = 8;
  private static final int FINISHED = 9;
  private static final int CLOSED = 10;
  private static final int INIT_WRITE = 11;
  private static final int WRITE = 12;

  private static final int DEFAULT_CODE_LENGTH = 8;
  private static final int CODE_LENGTH_REPEAT_CODE = 16;
  private static final int NUM_LITERAL_CODES = 256;
  private static final int NUM_INSERT_AND_COPY_CODES = 704;
  private static final int NUM_BLOCK_LENGTH_CODES = 26;
  private static final int LITERAL_CONTEXT_BITS = 6;
  private static final int DISTANCE_CONTEXT_BITS = 2;

  private static final int HUFFMAN_TABLE_BITS = 8;
  private static final int HUFFMAN_TABLE_MASK = 0xFF;

  /**
   * Maximum possible Huffman table size for an alphabet size of 704, max code length 15 and root
   * table bits 8.
   */
  static final int HUFFMAN_TABLE_SIZE = 1080;

  private static final int CODE_LENGTH_CODES = 18;
  private static final int[] CODE_LENGTH_CODE_ORDER = {
      1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  };

  private static final int NUM_DISTANCE_SHORT_CODES = 16;
  private static final int[] DISTANCE_SHORT_CODE_INDEX_OFFSET = {
      3, 2, 1, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2
  };

  private static final int[] DISTANCE_SHORT_CODE_VALUE_OFFSET = {
      0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3
  };

  /**
   * Static Huffman code for the code length code lengths.
   */
  private static final int[] FIXED_TABLE = {
      0x020000, 0x020004, 0x020003, 0x030002, 0x020000, 0x020004, 0x020003, 0x040001,
      0x020000, 0x020004, 0x020003, 0x030002, 0x020000, 0x020004, 0x020003, 0x040005
  };

  static final int[] DICTIONARY_OFFSETS_BY_LENGTH = {
    0, 0, 0, 0, 0, 4096, 9216, 21504, 35840, 44032, 53248, 63488, 74752, 87040, 93696, 100864,
    104704, 106752, 108928, 113536, 115968, 118528, 119872, 121280, 122016
  };

  static final int[] DICTIONARY_SIZE_BITS_BY_LENGTH = {
    0, 0, 0, 0, 10, 10, 11, 11, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 8, 7, 7, 6, 6, 5, 5
  };

  static final int MIN_WORD_LENGTH = 4;

  static final int MAX_WORD_LENGTH = 24;

  static final int MAX_TRANSFORMED_WORD_LENGTH = 5 + MAX_WORD_LENGTH + 8;

  //----------------------------------------------------------------------------
  // Prefix code LUT.
  //----------------------------------------------------------------------------
  static final int[] BLOCK_LENGTH_OFFSET = {
      1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305, 369, 497,
      753, 1265, 2289, 4337, 8433, 16625
  };

  static final int[] BLOCK_LENGTH_N_BITS = {
      2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24
  };

  static final int[] INSERT_LENGTH_OFFSET = {
      0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210,
      22594
  };

  static final int[] INSERT_LENGTH_N_BITS = {
      0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24
  };

  static final int[] COPY_LENGTH_OFFSET = {
      2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326, 582, 1094,
      2118
  };

  static final int[] COPY_LENGTH_N_BITS = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24
  };

  static final int[] INSERT_RANGE_LUT = {
      0, 0, 8, 8, 0, 16, 8, 16, 16
  };

  static final int[] COPY_RANGE_LUT = {
      0, 8, 0, 8, 16, 0, 16, 8, 16
  };

  private static int decodeWindowBits(State s) {
    BitReader.fillBitWindow(s);
    if (BitReader.readFewBits(s, 1) == 0) {
      return 16;
    }
    int n = BitReader.readFewBits(s, 3);
    if (n != 0) {
      return 17 + n;
    }
    n = BitReader.readFewBits(s, 3);
    if (n != 0) {
      return 8 + n;
    }
    return 17;
  }

  /**
   * Associate input with decoder state.
   *
   * @param s uninitialized state without associated input
   * @param input compressed data source
   */
  static void initState(State s, InputStream input) {
    if (s.runningState != UNINITIALIZED) {
      throw new IllegalStateException("State MUST be uninitialized");
    }
    s.blockTrees = new int[6 * HUFFMAN_TABLE_SIZE];
    s.input = input;
    BitReader.initBitReader(s);
    int windowBits = decodeWindowBits(s);
    if (windowBits == 9) { /* Reserved case for future expansion. */
      throw new BrotliRuntimeException("Invalid 'windowBits' code");
    }
    s.maxRingBufferSize = 1 << windowBits;
    s.maxBackwardDistance = s.maxRingBufferSize - 16;
    s.runningState = BLOCK_START;
  }

  static void close(State s) throws IOException {
    if (s.runningState == UNINITIALIZED) {
      throw new IllegalStateException("State MUST be initialized");
    }
    if (s.runningState == CLOSED) {
      return;
    }
    s.runningState = CLOSED;
    if (s.input != null) {
      Utils.closeInput(s.input);
      s.input = null;
    }
  }

  /**
   * Decodes a number in the range [0..255], by reading 1 - 11 bits.
   */
  private static int decodeVarLenUnsignedByte(State s) {
    BitReader.fillBitWindow(s);
    if (BitReader.readFewBits(s, 1) != 0) {
      int n = BitReader.readFewBits(s, 3);
      if (n == 0) {
        return 1;
      } else {
        return BitReader.readFewBits(s, n) + (1 << n);
      }
    }
    return 0;
  }

  private static void decodeMetaBlockLength(State s) {
    BitReader.fillBitWindow(s);
    s.inputEnd = BitReader.readFewBits(s, 1);
    s.metaBlockLength = 0;
    s.isUncompressed = 0;
    s.isMetadata = 0;
    if ((s.inputEnd != 0) && BitReader.readFewBits(s, 1) != 0) {
      return;
    }
    int sizeNibbles = BitReader.readFewBits(s, 2) + 4;
    if (sizeNibbles == 7) {
      s.isMetadata = 1;
      if (BitReader.readFewBits(s, 1) != 0) {
        throw new BrotliRuntimeException("Corrupted reserved bit");
      }
      int sizeBytes = BitReader.readFewBits(s, 2);
      if (sizeBytes == 0) {
        return;
      }
      for (int i = 0; i < sizeBytes; i++) {
        BitReader.fillBitWindow(s);
        int bits = BitReader.readFewBits(s, 8);
        if (bits == 0 && i + 1 == sizeBytes && sizeBytes > 1) {
          throw new BrotliRuntimeException("Exuberant nibble");
        }
        s.metaBlockLength |= bits << (i * 8);
      }
    } else {
      for (int i = 0; i < sizeNibbles; i++) {
        BitReader.fillBitWindow(s);
        int bits = BitReader.readFewBits(s, 4);
        if (bits == 0 && i + 1 == sizeNibbles && sizeNibbles > 4) {
          throw new BrotliRuntimeException("Exuberant nibble");
        }
        s.metaBlockLength |= bits << (i * 4);
      }
    }
    s.metaBlockLength++;
    if (s.inputEnd == 0) {
      s.isUncompressed = BitReader.readFewBits(s, 1);
    }
  }

  /**
   * Decodes the next Huffman code from bit-stream.
   */
  private static int readSymbol(int[] table, int offset, State s) {
    int val = BitReader.peekBits(s);
    offset += val & HUFFMAN_TABLE_MASK;
    int bits = table[offset] >> 16;
    int sym = table[offset] & 0xFFFF;
    if (bits <= HUFFMAN_TABLE_BITS) {
      s.bitOffset += bits;
      return sym;
    }
    offset += sym;
    int mask = (1 << bits) - 1;
    offset += (val & mask) >>> HUFFMAN_TABLE_BITS;
    s.bitOffset += ((table[offset] >> 16) + HUFFMAN_TABLE_BITS);
    return table[offset] & 0xFFFF;
  }

  private static int readBlockLength(int[] table, int offset, State s) {
    BitReader.fillBitWindow(s);
    int code = readSymbol(table, offset, s);
    int n = BLOCK_LENGTH_N_BITS[code];
    BitReader.fillBitWindow(s);
    return BLOCK_LENGTH_OFFSET[code] + BitReader.readBits(s, n);
  }

  private static int translateShortCodes(int code, int[] ringBuffer, int index) {
    if (code < NUM_DISTANCE_SHORT_CODES) {
      index += DISTANCE_SHORT_CODE_INDEX_OFFSET[code];
      index &= 3;
      return ringBuffer[index] + DISTANCE_SHORT_CODE_VALUE_OFFSET[code];
    }
    return code - NUM_DISTANCE_SHORT_CODES + 1;
  }

  private static void moveToFront(int[] v, int index) {
    int value = v[index];
    for (; index > 0; index--) {
      v[index] = v[index - 1];
    }
    v[0] = value;
  }

  private static void inverseMoveToFrontTransform(byte[] v, int vLen) {
    int[] mtf = new int[256];
    for (int i = 0; i < 256; i++) {
      mtf[i] = i;
    }
    for (int i = 0; i < vLen; i++) {
      int index = v[i] & 0xFF;
      v[i] = (byte) mtf[index];
      if (index != 0) {
        moveToFront(mtf, index);
      }
    }
  }

  private static void readHuffmanCodeLengths(
      int[] codeLengthCodeLengths, int numSymbols, int[] codeLengths, State s) {
    int symbol = 0;
    int prevCodeLen = DEFAULT_CODE_LENGTH;
    int repeat = 0;
    int repeatCodeLen = 0;
    int space = 32768;
    int[] table = new int[32];

    Huffman.buildHuffmanTable(table, 0, 5, codeLengthCodeLengths, CODE_LENGTH_CODES);

    while (symbol < numSymbols && space > 0) {
      BitReader.readMoreInput(s);
      BitReader.fillBitWindow(s);
      int p = BitReader.peekBits(s) & 31;
      s.bitOffset += table[p] >> 16;
      int codeLen = table[p] & 0xFFFF;
      if (codeLen < CODE_LENGTH_REPEAT_CODE) {
        repeat = 0;
        codeLengths[symbol++] = codeLen;
        if (codeLen != 0) {
          prevCodeLen = codeLen;
          space -= 32768 >> codeLen;
        }
      } else {
        int extraBits = codeLen - 14;
        int newLen = 0;
        if (codeLen == CODE_LENGTH_REPEAT_CODE) {
          newLen = prevCodeLen;
        }
        if (repeatCodeLen != newLen) {
          repeat = 0;
          repeatCodeLen = newLen;
        }
        int oldRepeat = repeat;
        if (repeat > 0) {
          repeat -= 2;
          repeat <<= extraBits;
        }
        BitReader.fillBitWindow(s);
        repeat += BitReader.readFewBits(s, extraBits) + 3;
        int repeatDelta = repeat - oldRepeat;
        if (symbol + repeatDelta > numSymbols) {
          throw new BrotliRuntimeException("symbol + repeatDelta > numSymbols"); // COV_NF_LINE
        }
        for (int i = 0; i < repeatDelta; i++) {
          codeLengths[symbol++] = repeatCodeLen;
        }
        if (repeatCodeLen != 0) {
          space -= repeatDelta << (15 - repeatCodeLen);
        }
      }
    }
    if (space != 0) {
      throw new BrotliRuntimeException("Unused space"); // COV_NF_LINE
    }
    // TODO: Pass max_symbol to Huffman table builder instead?
    Utils.fillIntsWithZeroes(codeLengths, symbol, numSymbols);
  }

  static int checkDupes(int[] symbols, int length) {
    for (int i = 0; i < length - 1; ++i) {
      for (int j = i + 1; j < length; ++j) {
        if (symbols[i] == symbols[j]) {
          return 0;
        }
      }
    }
    return 1;
  }

  // TODO: Use specialized versions for smaller tables.
  static void readHuffmanCode(int alphabetSize, int[] table, int offset, State s) {
    int ok = 1;
    int simpleCodeOrSkip;
    BitReader.readMoreInput(s);
    // TODO: Avoid allocation.
    int[] codeLengths = new int[alphabetSize];
    BitReader.fillBitWindow(s);
    simpleCodeOrSkip = BitReader.readFewBits(s, 2);
    if (simpleCodeOrSkip == 1) { // Read symbols, codes & code lengths directly.
      int maxBitsCounter = alphabetSize - 1;
      int maxBits = 0;
      int[] symbols = new int[4];
      int numSymbols = BitReader.readFewBits(s, 2) + 1;
      while (maxBitsCounter != 0) {
        maxBitsCounter >>= 1;
        maxBits++;
      }
      // TODO: uncomment when codeLengths is reused.
      // Utils.fillWithZeroes(codeLengths, 0, alphabetSize);
      for (int i = 0; i < numSymbols; i++) {
        BitReader.fillBitWindow(s);
        symbols[i] = BitReader.readFewBits(s, maxBits) % alphabetSize;
        codeLengths[symbols[i]] = 2;
      }
      codeLengths[symbols[0]] = 1;
      switch (numSymbols) {
        case 2:
          codeLengths[symbols[1]] = 1;
          break;
        case 4:
          if (BitReader.readFewBits(s, 1) == 1) {
            codeLengths[symbols[2]] = 3;
            codeLengths[symbols[3]] = 3;
          } else {
            codeLengths[symbols[0]] = 2;
          }
          break;
        default:
          break;
      }
      ok = checkDupes(symbols, numSymbols);
    } else { // Decode Huffman-coded code lengths.
      int[] codeLengthCodeLengths = new int[CODE_LENGTH_CODES];
      int space = 32;
      int numCodes = 0;
      for (int i = simpleCodeOrSkip; i < CODE_LENGTH_CODES && space > 0; i++) {
        int codeLenIdx = CODE_LENGTH_CODE_ORDER[i];
        BitReader.fillBitWindow(s);
        int p = BitReader.peekBits(s) & 15;
        // TODO: Demultiplex FIXED_TABLE.
        s.bitOffset += FIXED_TABLE[p] >> 16;
        int v = FIXED_TABLE[p] & 0xFFFF;
        codeLengthCodeLengths[codeLenIdx] = v;
        if (v != 0) {
          space -= (32 >> v);
          numCodes++;
        }
      }
      if (space != 0 && numCodes != 1) {
        ok = 0;
      }
      readHuffmanCodeLengths(codeLengthCodeLengths, alphabetSize, codeLengths, s);
    }
    if (ok == 0) {
      throw new BrotliRuntimeException("Can't readHuffmanCode"); // COV_NF_LINE
    }
    Huffman.buildHuffmanTable(table, offset, HUFFMAN_TABLE_BITS, codeLengths, alphabetSize);
  }

  private static int decodeContextMap(int contextMapSize, byte[] contextMap, State s) {
    BitReader.readMoreInput(s);
    int numTrees = decodeVarLenUnsignedByte(s) + 1;

    if (numTrees == 1) {
      Utils.fillBytesWithZeroes(contextMap, 0, contextMapSize);
      return numTrees;
    }

    BitReader.fillBitWindow(s);
    int useRleForZeros = BitReader.readFewBits(s, 1);
    int maxRunLengthPrefix = 0;
    if (useRleForZeros != 0) {
      maxRunLengthPrefix = BitReader.readFewBits(s, 4) + 1;
    }
    int[] table = new int[HUFFMAN_TABLE_SIZE];
    readHuffmanCode(numTrees + maxRunLengthPrefix, table, 0, s);
    for (int i = 0; i < contextMapSize; ) {
      BitReader.readMoreInput(s);
      BitReader.fillBitWindow(s);
      int code = readSymbol(table, 0, s);
      if (code == 0) {
        contextMap[i] = 0;
        i++;
      } else if (code <= maxRunLengthPrefix) {
        BitReader.fillBitWindow(s);
        int reps = (1 << code) + BitReader.readFewBits(s, code);
        while (reps != 0) {
          if (i >= contextMapSize) {
            throw new BrotliRuntimeException("Corrupted context map"); // COV_NF_LINE
          }
          contextMap[i] = 0;
          i++;
          reps--;
        }
      } else {
        contextMap[i] = (byte) (code - maxRunLengthPrefix);
        i++;
      }
    }
    BitReader.fillBitWindow(s);
    if (BitReader.readFewBits(s, 1) == 1) {
      inverseMoveToFrontTransform(contextMap, contextMapSize);
    }
    return numTrees;
  }

  private static int decodeBlockTypeAndLength(State s, int treeType, int numBlockTypes) {
    final int[] ringBuffers = s.rings;
    final int offset = 4 + treeType * 2;
    BitReader.fillBitWindow(s);
    int blockType = readSymbol(s.blockTrees, treeType * HUFFMAN_TABLE_SIZE, s);
    int result = readBlockLength(s.blockTrees, (treeType + 3) * HUFFMAN_TABLE_SIZE, s);

    if (blockType == 1) {
      blockType = ringBuffers[offset + 1] + 1;
    } else if (blockType == 0) {
      blockType = ringBuffers[offset];
    } else {
      blockType -= 2;
    }
    if (blockType >= numBlockTypes) {
      blockType -= numBlockTypes;
    }
    ringBuffers[offset] = ringBuffers[offset + 1];
    ringBuffers[offset + 1] = blockType;
    return result;
  }

  private static void decodeLiteralBlockSwitch(State s) {
    s.literalBlockLength = decodeBlockTypeAndLength(s, 0, s.numLiteralBlockTypes);
    int literalBlockType = s.rings[5];
    s.contextMapSlice = literalBlockType << LITERAL_CONTEXT_BITS;
    s.literalTreeIndex = s.contextMap[s.contextMapSlice] & 0xFF;
    s.literalTree = s.hGroup0[s.literalTreeIndex];
    int contextMode = s.contextModes[literalBlockType];
    s.contextLookupOffset1 = contextMode << 9;
    s.contextLookupOffset2 = s.contextLookupOffset1 + 256;
  }

  private static void decodeCommandBlockSwitch(State s) {
    s.commandBlockLength = decodeBlockTypeAndLength(s, 1, s.numCommandBlockTypes);
    s.treeCommandOffset = s.hGroup1[s.rings[7]];
  }

  private static void decodeDistanceBlockSwitch(State s) {
    s.distanceBlockLength = decodeBlockTypeAndLength(s, 2, s.numDistanceBlockTypes);
    s.distContextMapSlice = s.rings[9] << DISTANCE_CONTEXT_BITS;
  }

  private static void maybeReallocateRingBuffer(State s) {
    int newSize = s.maxRingBufferSize;
    if (newSize > s.expectedTotalSize) {
      /* TODO: Handle 2GB+ cases more gracefully. */
      int minimalNewSize = s.expectedTotalSize;
      while ((newSize >> 1) > minimalNewSize) {
        newSize >>= 1;
      }
      if ((s.inputEnd == 0) && newSize < 16384 && s.maxRingBufferSize >= 16384) {
        newSize = 16384;
      }
    }
    if (newSize <= s.ringBufferSize) {
      return;
    }
    int ringBufferSizeWithSlack = newSize + MAX_TRANSFORMED_WORD_LENGTH;
    byte[] newBuffer = new byte[ringBufferSizeWithSlack];
    if (s.ringBuffer.length != 0) {
      System.arraycopy(s.ringBuffer, 0, newBuffer, 0, s.ringBufferSize);
    }
    s.ringBuffer = newBuffer;
    s.ringBufferSize = newSize;
  }

  private static void readNextMetablockHeader(State s) {
    if (s.inputEnd != 0) {
      s.nextRunningState = FINISHED;
      s.runningState = INIT_WRITE;
      return;
    }
    // TODO: Reset? Do we need this?
    s.hGroup0 = new int[0];
    s.hGroup1 = new int[0];
    s.hGroup2 = new int[0];

    BitReader.readMoreInput(s);
    decodeMetaBlockLength(s);
    if ((s.metaBlockLength == 0) && (s.isMetadata == 0)) {
      return;
    }
    if ((s.isUncompressed != 0) || (s.isMetadata != 0)) {
      BitReader.jumpToByteBoundary(s);
      s.runningState = (s.isMetadata != 0) ? READ_METADATA : COPY_UNCOMPRESSED;
    } else {
      s.runningState = COMPRESSED_BLOCK_START;
    }

    if (s.isMetadata != 0) {
      return;
    }
    s.expectedTotalSize += s.metaBlockLength;
    if (s.expectedTotalSize > 1 << 30) {
      s.expectedTotalSize = 1 << 30;
    }
    if (s.ringBufferSize < s.maxRingBufferSize) {
      maybeReallocateRingBuffer(s);
    }
  }

  private static int readMetablockPartition(State s, int treeType, int numBlockTypes) {
    if (numBlockTypes <= 1) {
      return 1 << 28;
    }
    readHuffmanCode(numBlockTypes + 2, s.blockTrees, treeType * HUFFMAN_TABLE_SIZE, s);
    readHuffmanCode(NUM_BLOCK_LENGTH_CODES, s.blockTrees, (treeType + 3) * HUFFMAN_TABLE_SIZE, s);
    return readBlockLength(s.blockTrees, (treeType + 3) * HUFFMAN_TABLE_SIZE, s);
  }

  private static void readMetablockHuffmanCodesAndContextMaps(State s) {
    s.numLiteralBlockTypes = decodeVarLenUnsignedByte(s) + 1;
    s.literalBlockLength = readMetablockPartition(s, 0, s.numLiteralBlockTypes);
    s.numCommandBlockTypes = decodeVarLenUnsignedByte(s) + 1;
    s.commandBlockLength = readMetablockPartition(s, 1, s.numCommandBlockTypes);
    s.numDistanceBlockTypes = decodeVarLenUnsignedByte(s) + 1;
    s.distanceBlockLength = readMetablockPartition(s, 2, s.numDistanceBlockTypes);

    BitReader.readMoreInput(s);
    BitReader.fillBitWindow(s);
    s.distancePostfixBits = BitReader.readFewBits(s, 2);
    s.numDirectDistanceCodes =
        NUM_DISTANCE_SHORT_CODES + (BitReader.readFewBits(s, 4) << s.distancePostfixBits);
    s.distancePostfixMask = (1 << s.distancePostfixBits) - 1;
    int numDistanceCodes = s.numDirectDistanceCodes + (48 << s.distancePostfixBits);
    // TODO: Reuse?
    s.contextModes = new byte[s.numLiteralBlockTypes];
    for (int i = 0; i < s.numLiteralBlockTypes;) {
      /* Ensure that less than 256 bits read between readMoreInput. */
      int limit = Math.min(i + 96, s.numLiteralBlockTypes);
      for (; i < limit; ++i) {
        BitReader.fillBitWindow(s);
        s.contextModes[i] = (byte) (BitReader.readFewBits(s, 2));
      }
      BitReader.readMoreInput(s);
    }

    // TODO: Reuse?
    s.contextMap = new byte[s.numLiteralBlockTypes << LITERAL_CONTEXT_BITS];
    int numLiteralTrees = decodeContextMap(s.numLiteralBlockTypes << LITERAL_CONTEXT_BITS,
        s.contextMap, s);
    s.trivialLiteralContext = 1;
    for (int j = 0; j < s.numLiteralBlockTypes << LITERAL_CONTEXT_BITS; j++) {
      if (s.contextMap[j] != j >> LITERAL_CONTEXT_BITS) {
        s.trivialLiteralContext = 0;
        break;
      }
    }

    // TODO: Reuse?
    s.distContextMap = new byte[s.numDistanceBlockTypes << DISTANCE_CONTEXT_BITS];
    int numDistTrees = decodeContextMap(s.numDistanceBlockTypes << DISTANCE_CONTEXT_BITS,
        s.distContextMap, s);

    s.hGroup0 = decodeHuffmanTreeGroup(NUM_LITERAL_CODES, numLiteralTrees, s);
    s.hGroup1 =
        decodeHuffmanTreeGroup(NUM_INSERT_AND_COPY_CODES, s.numCommandBlockTypes, s);
    s.hGroup2 = decodeHuffmanTreeGroup(numDistanceCodes, numDistTrees, s);

    s.contextMapSlice = 0;
    s.distContextMapSlice = 0;
    s.contextLookupOffset1 = (int) (s.contextModes[0]) << 9;
    s.contextLookupOffset2 = s.contextLookupOffset1 + 256;
    s.literalTreeIndex = 0;
    s.literalTree = s.hGroup0[0];
    s.treeCommandOffset = s.hGroup1[0];

    s.rings[4] = 1;
    s.rings[5] = 0;
    s.rings[6] = 1;
    s.rings[7] = 0;
    s.rings[8] = 1;
    s.rings[9] = 0;
  }

  private static void copyUncompressedData(State s) {
    final byte[] ringBuffer = s.ringBuffer;

    // Could happen if block ends at ring buffer end.
    if (s.metaBlockLength <= 0) {
      BitReader.reload(s);
      s.runningState = BLOCK_START;
      return;
    }

    int chunkLength = Math.min(s.ringBufferSize - s.pos, s.metaBlockLength);
    BitReader.copyBytes(s, ringBuffer, s.pos, chunkLength);
    s.metaBlockLength -= chunkLength;
    s.pos += chunkLength;
    if (s.pos == s.ringBufferSize) {
        s.nextRunningState = COPY_UNCOMPRESSED;
        s.runningState = INIT_WRITE;
        return;
      }

    BitReader.reload(s);
    s.runningState = BLOCK_START;
  }

  private static int writeRingBuffer(State s) {
    int toWrite = Math.min(s.outputLength - s.outputUsed,
        s.ringBufferBytesReady - s.ringBufferBytesWritten);
    if (toWrite != 0) {
      System.arraycopy(s.ringBuffer, s.ringBufferBytesWritten, s.output,
          s.outputOffset + s.outputUsed, toWrite);
      s.outputUsed += toWrite;
      s.ringBufferBytesWritten += toWrite;
    }

    if (s.outputUsed < s.outputLength) {
      return 1;
    } else {
      return 0;
    }
  }

  private static int[] decodeHuffmanTreeGroup(int alphabetSize, int n, State s) {
    int[] group = new int[n + (n * HUFFMAN_TABLE_SIZE)];
    int next = n;
    for (int i = 0; i < n; i++) {
      group[i] = next;
      Decode.readHuffmanCode(alphabetSize, group, next, s);
      next += HUFFMAN_TABLE_SIZE;
    }
    return group;
  }

  // Returns offset in ringBuffer that should trigger WRITE when filled.
  private static int calculateFence(State s) {
    int result = s.ringBufferSize;
    if (s.isEager != 0) {
      result = Math.min(result, s.ringBufferBytesWritten + s.outputLength - s.outputUsed);
    }
    return result;
  }

  /**
   * Actual decompress implementation.
   */
  static void decompress(State s) {
    if (s.runningState == UNINITIALIZED) {
      throw new IllegalStateException("Can't decompress until initialized");
    }
    if (s.runningState == CLOSED) {
      throw new IllegalStateException("Can't decompress after close");
    }
    int fence = calculateFence(s);
    int ringBufferMask = s.ringBufferSize - 1;
    byte[] ringBuffer = s.ringBuffer;

    while (s.runningState != FINISHED) {
      // TODO: extract cases to methods for the better readability.
      switch (s.runningState) {
        case BLOCK_START:
          if (s.metaBlockLength < 0) {
            throw new BrotliRuntimeException("Invalid metablock length");
          }
          readNextMetablockHeader(s);
          /* Ring-buffer would be reallocated here. */
          fence = calculateFence(s);
          ringBufferMask = s.ringBufferSize - 1;
          ringBuffer = s.ringBuffer;
          continue;

        case COMPRESSED_BLOCK_START:
          readMetablockHuffmanCodesAndContextMaps(s);
          s.runningState = MAIN_LOOP;
          // Fall through

        case MAIN_LOOP:
          if (s.metaBlockLength <= 0) {
            s.runningState = BLOCK_START;
            continue;
          }
          BitReader.readMoreInput(s);
          if (s.commandBlockLength == 0) {
            decodeCommandBlockSwitch(s);
          }
          s.commandBlockLength--;
          BitReader.fillBitWindow(s);
          int cmdCode = readSymbol(s.hGroup1, s.treeCommandOffset, s);
          int rangeIdx = cmdCode >>> 6;
          s.distanceCode = 0;
          if (rangeIdx >= 2) {
            rangeIdx -= 2;
            s.distanceCode = -1;
          }
          int insertCode = INSERT_RANGE_LUT[rangeIdx] + ((cmdCode >>> 3) & 7);
          BitReader.fillBitWindow(s);
          int insertBits = INSERT_LENGTH_N_BITS[insertCode];
          int insertExtra = BitReader.readBits(s, insertBits);
          s.insertLength = INSERT_LENGTH_OFFSET[insertCode] + insertExtra;
          int copyCode = COPY_RANGE_LUT[rangeIdx] + (cmdCode & 7);
          BitReader.fillBitWindow(s);
          int copyBits = COPY_LENGTH_N_BITS[copyCode];
          int copyExtra = BitReader.readBits(s, copyBits);
          s.copyLength = COPY_LENGTH_OFFSET[copyCode] + copyExtra;

          s.j = 0;
          s.runningState = INSERT_LOOP;

          // Fall through
        case INSERT_LOOP:
          if (s.trivialLiteralContext != 0) {
            while (s.j < s.insertLength) {
              BitReader.readMoreInput(s);
              if (s.literalBlockLength == 0) {
                decodeLiteralBlockSwitch(s);
              }
              s.literalBlockLength--;
              BitReader.fillBitWindow(s);
              ringBuffer[s.pos] =
                  (byte) readSymbol(s.hGroup0, s.literalTree, s);
              s.pos++;
              s.j++;
              if (s.pos >= fence) {
                s.nextRunningState = INSERT_LOOP;
                s.runningState = INIT_WRITE;
                break;
              }
            }
          } else {
            int prevByte1 = ringBuffer[(s.pos - 1) & ringBufferMask] & 0xFF;
            int prevByte2 = ringBuffer[(s.pos - 2) & ringBufferMask] & 0xFF;
            while (s.j < s.insertLength) {
              BitReader.readMoreInput(s);
              if (s.literalBlockLength == 0) {
                decodeLiteralBlockSwitch(s);
              }
              int literalTreeIndex = s.contextMap[s.contextMapSlice
                + (Context.LOOKUP[s.contextLookupOffset1 + prevByte1]
                    | Context.LOOKUP[s.contextLookupOffset2 + prevByte2])] & 0xFF;
              s.literalBlockLength--;
              prevByte2 = prevByte1;
              BitReader.fillBitWindow(s);
              prevByte1 = readSymbol(
                  s.hGroup0, s.hGroup0[literalTreeIndex], s);
              ringBuffer[s.pos] = (byte) prevByte1;
              s.pos++;
              s.j++;
              if (s.pos >= fence) {
                s.nextRunningState = INSERT_LOOP;
                s.runningState = INIT_WRITE;
                break;
              }
            }
          }
          if (s.runningState != INSERT_LOOP) {
            continue;
          }
          s.metaBlockLength -= s.insertLength;
          if (s.metaBlockLength <= 0) {
            s.runningState = MAIN_LOOP;
            continue;
          }
          if (s.distanceCode < 0) {
            BitReader.readMoreInput(s);
            if (s.distanceBlockLength == 0) {
              decodeDistanceBlockSwitch(s);
            }
            s.distanceBlockLength--;
            BitReader.fillBitWindow(s);
            s.distanceCode = readSymbol(s.hGroup2, s.hGroup2[
                s.distContextMap[s.distContextMapSlice
                    + (s.copyLength > 4 ? 3 : s.copyLength - 2)] & 0xFF], s);
            if (s.distanceCode >= s.numDirectDistanceCodes) {
              s.distanceCode -= s.numDirectDistanceCodes;
              int postfix = s.distanceCode & s.distancePostfixMask;
              s.distanceCode >>>= s.distancePostfixBits;
              int n = (s.distanceCode >>> 1) + 1;
              int offset = ((2 + (s.distanceCode & 1)) << n) - 4;
              BitReader.fillBitWindow(s);
              int distanceExtra = BitReader.readBits(s, n);
              s.distanceCode = s.numDirectDistanceCodes + postfix
                  + ((offset + distanceExtra) << s.distancePostfixBits);
            }
          }

          // Convert the distance code to the actual distance by possibly looking up past distances
          // from the ringBuffer.
          s.distance = translateShortCodes(s.distanceCode, s.rings, s.distRbIdx);
          if (s.distance < 0) {
            throw new BrotliRuntimeException("Negative distance"); // COV_NF_LINE
          }

          if (s.maxDistance != s.maxBackwardDistance
              && s.pos < s.maxBackwardDistance) {
            s.maxDistance = s.pos;
          } else {
            s.maxDistance = s.maxBackwardDistance;
          }

          if (s.distance > s.maxDistance) {
            s.runningState = TRANSFORM;
            continue;
          }

          if (s.distanceCode > 0) {
            s.rings[s.distRbIdx & 3] = s.distance;
            s.distRbIdx++;
          }

          if (s.copyLength > s.metaBlockLength) {
            throw new BrotliRuntimeException("Invalid backward reference"); // COV_NF_LINE
          }
          s.j = 0;
          s.runningState = COPY_LOOP;
          // fall through
        case COPY_LOOP:
          int src = (s.pos - s.distance) & ringBufferMask;
          int dst = s.pos;
          int copyLength = s.copyLength - s.j;
          int srcEnd = src + copyLength;
          int dstEnd = dst + copyLength;
          if ((srcEnd < ringBufferMask) && (dstEnd < ringBufferMask)) {
            if (copyLength < 12 || (srcEnd > dst && dstEnd > src)) {
              for (int k = 0; k < copyLength; ++k) {
                ringBuffer[dst++] = ringBuffer[src++];
              }
            } else {
              Utils.copyBytesWithin(ringBuffer, dst, src, srcEnd);
            }
            s.j += copyLength;
            s.metaBlockLength -= copyLength;
            s.pos += copyLength;
          } else {
            for (; s.j < s.copyLength;) {
              ringBuffer[s.pos] =
                  ringBuffer[(s.pos - s.distance) & ringBufferMask];
              s.metaBlockLength--;
              s.pos++;
              s.j++;
              if (s.pos >= fence) {
                s.nextRunningState = COPY_LOOP;
                s.runningState = INIT_WRITE;
                break;
              }
            }
          }
          if (s.runningState == COPY_LOOP) {
            s.runningState = MAIN_LOOP;
          }
          continue;

        case TRANSFORM:
          if (s.copyLength >= MIN_WORD_LENGTH
              && s.copyLength <= MAX_WORD_LENGTH) {
            int offset = DICTIONARY_OFFSETS_BY_LENGTH[s.copyLength];
            int wordId = s.distance - s.maxDistance - 1;
            int shift = DICTIONARY_SIZE_BITS_BY_LENGTH[s.copyLength];
            int mask = (1 << shift) - 1;
            int wordIdx = wordId & mask;
            int transformIdx = wordId >>> shift;
            offset += wordIdx * s.copyLength;
            if (transformIdx < Transform.NUM_TRANSFORMS) {
              int len = Transform.transformDictionaryWord(ringBuffer, s.pos,
                  Dictionary.getData(), offset, s.copyLength, transformIdx);
              s.pos += len;
              s.metaBlockLength -= len;
              if (s.pos >= fence) {
                s.nextRunningState = MAIN_LOOP;
                s.runningState = INIT_WRITE;
                continue;
              }
            } else {
              throw new BrotliRuntimeException("Invalid backward reference"); // COV_NF_LINE
            }
          } else {
            throw new BrotliRuntimeException("Invalid backward reference"); // COV_NF_LINE
          }
          s.runningState = MAIN_LOOP;
          continue;

        case READ_METADATA:
          while (s.metaBlockLength > 0) {
            BitReader.readMoreInput(s);
            // Optimize
            BitReader.fillBitWindow(s);
            BitReader.readFewBits(s, 8);
            s.metaBlockLength--;
          }
          s.runningState = BLOCK_START;
          continue;


        case COPY_UNCOMPRESSED:
          copyUncompressedData(s);
          continue;

        case INIT_WRITE:
          s.ringBufferBytesReady = Math.min(s.pos, s.ringBufferSize);
          s.runningState = WRITE;
          // fall through
        case WRITE:
          if (writeRingBuffer(s) == 0) {
            // Output buffer is full.
            return;
          }
          if (s.pos >= s.maxBackwardDistance) {
            s.maxDistance = s.maxBackwardDistance;
          }
          // Wrap the ringBuffer.
          if (s.pos >= s.ringBufferSize) {
            if (s.pos > s.ringBufferSize) {
              Utils.copyBytesWithin(ringBuffer, 0, s.ringBufferSize, s.pos);
            }
            s.pos &= ringBufferMask;
            s.ringBufferBytesWritten = 0;
          }
          s.runningState = s.nextRunningState;
          continue;

        default:
          throw new BrotliRuntimeException("Unexpected state " + s.runningState);
      }
    }
    if (s.runningState == FINISHED) {
      if (s.metaBlockLength < 0) {
        throw new BrotliRuntimeException("Invalid metablock length");
      }
      BitReader.jumpToByteBoundary(s);
      BitReader.checkHealth(s, 1);
    }
  }
}
