/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>API for Brotli decompression.</summary>
	internal sealed class Decode
	{
		private const int DefaultCodeLength = 8;

		private const int CodeLengthRepeatCode = 16;

		private const int NumLiteralCodes = 256;

		private const int NumInsertAndCopyCodes = 704;

		private const int NumBlockLengthCodes = 26;

		private const int LiteralContextBits = 6;

		private const int DistanceContextBits = 2;

		private const int HuffmanTableBits = 8;

		private const int HuffmanTableMask = unchecked((int)(0xFF));

		private const int CodeLengthCodes = 18;

		private static readonly int[] CodeLengthCodeOrder = new int[] { 1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

		private const int NumDistanceShortCodes = 16;

		private static readonly int[] DistanceShortCodeIndexOffset = new int[] { 3, 2, 1, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2 };

		private static readonly int[] DistanceShortCodeValueOffset = new int[] { 0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3, 3 };

		/// <summary>Static Huffman code for the code length code lengths.</summary>
		private static readonly int[] FixedTable = new int[] { unchecked((int)(0x020000)), unchecked((int)(0x020004)), unchecked((int)(0x020003)), unchecked((int)(0x030002)), unchecked((int)(0x020000)), unchecked((int)(0x020004)), unchecked((int)(0x020003
			)), unchecked((int)(0x040001)), unchecked((int)(0x020000)), unchecked((int)(0x020004)), unchecked((int)(0x020003)), unchecked((int)(0x030002)), unchecked((int)(0x020000)), unchecked((int)(0x020004)), unchecked((int)(0x020003)), unchecked((int
			)(0x040005)) };

		/// <summary>Decodes a number in the range [0..255], by reading 1 - 11 bits.</summary>
		private static int DecodeVarLenUnsignedByte(Org.Brotli.Dec.BitReader br)
		{
			if (Org.Brotli.Dec.BitReader.ReadBits(br, 1) != 0)
			{
				int n = Org.Brotli.Dec.BitReader.ReadBits(br, 3);
				if (n == 0)
				{
					return 1;
				}
				else
				{
					return Org.Brotli.Dec.BitReader.ReadBits(br, n) + (1 << n);
				}
			}
			return 0;
		}

		private static void DecodeMetaBlockLength(Org.Brotli.Dec.BitReader br, Org.Brotli.Dec.State state)
		{
			state.inputEnd = Org.Brotli.Dec.BitReader.ReadBits(br, 1) == 1;
			state.metaBlockLength = 0;
			state.isUncompressed = false;
			state.isMetadata = false;
			if (state.inputEnd && Org.Brotli.Dec.BitReader.ReadBits(br, 1) != 0)
			{
				return;
			}
			int sizeNibbles = Org.Brotli.Dec.BitReader.ReadBits(br, 2) + 4;
			if (sizeNibbles == 7)
			{
				state.isMetadata = true;
				if (Org.Brotli.Dec.BitReader.ReadBits(br, 1) != 0)
				{
					throw new Org.Brotli.Dec.BrotliRuntimeException("Corrupted reserved bit");
				}
				int sizeBytes = Org.Brotli.Dec.BitReader.ReadBits(br, 2);
				if (sizeBytes == 0)
				{
					return;
				}
				for (int i = 0; i < sizeBytes; i++)
				{
					int bits = Org.Brotli.Dec.BitReader.ReadBits(br, 8);
					if (bits == 0 && i + 1 == sizeBytes && sizeBytes > 1)
					{
						throw new Org.Brotli.Dec.BrotliRuntimeException("Exuberant nibble");
					}
					state.metaBlockLength |= bits << (i * 8);
				}
			}
			else
			{
				for (int i = 0; i < sizeNibbles; i++)
				{
					int bits = Org.Brotli.Dec.BitReader.ReadBits(br, 4);
					if (bits == 0 && i + 1 == sizeNibbles && sizeNibbles > 4)
					{
						throw new Org.Brotli.Dec.BrotliRuntimeException("Exuberant nibble");
					}
					state.metaBlockLength |= bits << (i * 4);
				}
			}
			state.metaBlockLength++;
			if (!state.inputEnd)
			{
				state.isUncompressed = Org.Brotli.Dec.BitReader.ReadBits(br, 1) == 1;
			}
		}

		/// <summary>Decodes the next Huffman code from bit-stream.</summary>
		private static int ReadSymbol(int[] table, int offset, Org.Brotli.Dec.BitReader br)
		{
			int val = (int)((long)(((ulong)br.accumulator) >> br.bitOffset));
			offset += val & HuffmanTableMask;
			int bits = table[offset] >> 16;
			int sym = table[offset] & unchecked((int)(0xFFFF));
			if (bits <= HuffmanTableBits)
			{
				br.bitOffset += bits;
				return sym;
			}
			offset += sym;
			int mask = (1 << bits) - 1;
			offset += (int)(((uint)(val & mask)) >> HuffmanTableBits);
			br.bitOffset += ((table[offset] >> 16) + HuffmanTableBits);
			return table[offset] & unchecked((int)(0xFFFF));
		}

		private static int ReadBlockLength(int[] table, int offset, Org.Brotli.Dec.BitReader br)
		{
			Org.Brotli.Dec.BitReader.FillBitWindow(br);
			int code = ReadSymbol(table, offset, br);
			int n = Org.Brotli.Dec.Prefix.BlockLengthNBits[code];
			return Org.Brotli.Dec.Prefix.BlockLengthOffset[code] + Org.Brotli.Dec.BitReader.ReadBits(br, n);
		}

		private static int TranslateShortCodes(int code, int[] ringBuffer, int index)
		{
			if (code < NumDistanceShortCodes)
			{
				index += DistanceShortCodeIndexOffset[code];
				index &= 3;
				return ringBuffer[index] + DistanceShortCodeValueOffset[code];
			}
			return code - NumDistanceShortCodes + 1;
		}

		private static void MoveToFront(int[] v, int index)
		{
			int value = v[index];
			for (; index > 0; index--)
			{
				v[index] = v[index - 1];
			}
			v[0] = value;
		}

		private static void InverseMoveToFrontTransform(byte[] v, int vLen)
		{
			int[] mtf = new int[256];
			for (int i = 0; i < 256; i++)
			{
				mtf[i] = i;
			}
			for (int i = 0; i < vLen; i++)
			{
				int index = v[i] & unchecked((int)(0xFF));
				v[i] = unchecked((byte)mtf[index]);
				if (index != 0)
				{
					MoveToFront(mtf, index);
				}
			}
		}

		private static void ReadHuffmanCodeLengths(int[] codeLengthCodeLengths, int numSymbols, int[] codeLengths, Org.Brotli.Dec.BitReader br)
		{
			int symbol = 0;
			int prevCodeLen = DefaultCodeLength;
			int repeat = 0;
			int repeatCodeLen = 0;
			int space = 32768;
			int[] table = new int[32];
			Org.Brotli.Dec.Huffman.BuildHuffmanTable(table, 0, 5, codeLengthCodeLengths, CodeLengthCodes);
			while (symbol < numSymbols && space > 0)
			{
				Org.Brotli.Dec.BitReader.ReadMoreInput(br);
				Org.Brotli.Dec.BitReader.FillBitWindow(br);
				int p = (int)(((long)(((ulong)br.accumulator) >> br.bitOffset))) & 31;
				br.bitOffset += table[p] >> 16;
				int codeLen = table[p] & unchecked((int)(0xFFFF));
				if (codeLen < CodeLengthRepeatCode)
				{
					repeat = 0;
					codeLengths[symbol++] = codeLen;
					if (codeLen != 0)
					{
						prevCodeLen = codeLen;
						space -= 32768 >> codeLen;
					}
				}
				else
				{
					int extraBits = codeLen - 14;
					int newLen = 0;
					if (codeLen == CodeLengthRepeatCode)
					{
						newLen = prevCodeLen;
					}
					if (repeatCodeLen != newLen)
					{
						repeat = 0;
						repeatCodeLen = newLen;
					}
					int oldRepeat = repeat;
					if (repeat > 0)
					{
						repeat -= 2;
						repeat <<= extraBits;
					}
					repeat += Org.Brotli.Dec.BitReader.ReadBits(br, extraBits) + 3;
					int repeatDelta = repeat - oldRepeat;
					if (symbol + repeatDelta > numSymbols)
					{
						throw new Org.Brotli.Dec.BrotliRuntimeException("symbol + repeatDelta > numSymbols");
					}
					// COV_NF_LINE
					for (int i = 0; i < repeatDelta; i++)
					{
						codeLengths[symbol++] = repeatCodeLen;
					}
					if (repeatCodeLen != 0)
					{
						space -= repeatDelta << (15 - repeatCodeLen);
					}
				}
			}
			if (space != 0)
			{
				throw new Org.Brotli.Dec.BrotliRuntimeException("Unused space");
			}
			// COV_NF_LINE
			// TODO: Pass max_symbol to Huffman table builder instead?
			Org.Brotli.Dec.Utils.FillWithZeroes(codeLengths, symbol, numSymbols - symbol);
		}

		// TODO: Use specialized versions for smaller tables.
		internal static void ReadHuffmanCode(int alphabetSize, int[] table, int offset, Org.Brotli.Dec.BitReader br)
		{
			bool ok = true;
			int simpleCodeOrSkip;
			Org.Brotli.Dec.BitReader.ReadMoreInput(br);
			// TODO: Avoid allocation.
			int[] codeLengths = new int[alphabetSize];
			simpleCodeOrSkip = Org.Brotli.Dec.BitReader.ReadBits(br, 2);
			if (simpleCodeOrSkip == 1)
			{
				// Read symbols, codes & code lengths directly.
				int maxBitsCounter = alphabetSize - 1;
				int maxBits = 0;
				int[] symbols = new int[4];
				int numSymbols = Org.Brotli.Dec.BitReader.ReadBits(br, 2) + 1;
				while (maxBitsCounter != 0)
				{
					maxBitsCounter >>= 1;
					maxBits++;
				}
				// TODO: uncomment when codeLengths is reused.
				// Utils.fillWithZeroes(codeLengths, 0, alphabetSize);
				for (int i = 0; i < numSymbols; i++)
				{
					symbols[i] = Org.Brotli.Dec.BitReader.ReadBits(br, maxBits) % alphabetSize;
					codeLengths[symbols[i]] = 2;
				}
				codeLengths[symbols[0]] = 1;
				switch (numSymbols)
				{
					case 1:
					{
						break;
					}

					case 2:
					{
						ok = symbols[0] != symbols[1];
						codeLengths[symbols[1]] = 1;
						break;
					}

					case 3:
					{
						ok = symbols[0] != symbols[1] && symbols[0] != symbols[2] && symbols[1] != symbols[2];
						break;
					}

					case 4:
					default:
					{
						ok = symbols[0] != symbols[1] && symbols[0] != symbols[2] && symbols[0] != symbols[3] && symbols[1] != symbols[2] && symbols[1] != symbols[3] && symbols[2] != symbols[3];
						if (Org.Brotli.Dec.BitReader.ReadBits(br, 1) == 1)
						{
							codeLengths[symbols[2]] = 3;
							codeLengths[symbols[3]] = 3;
						}
						else
						{
							codeLengths[symbols[0]] = 2;
						}
						break;
					}
				}
			}
			else
			{
				// Decode Huffman-coded code lengths.
				int[] codeLengthCodeLengths = new int[CodeLengthCodes];
				int space = 32;
				int numCodes = 0;
				for (int i = simpleCodeOrSkip; i < CodeLengthCodes && space > 0; i++)
				{
					int codeLenIdx = CodeLengthCodeOrder[i];
					Org.Brotli.Dec.BitReader.FillBitWindow(br);
					int p = (int)((long)(((ulong)br.accumulator) >> br.bitOffset)) & 15;
					// TODO: Demultiplex FIXED_TABLE.
					br.bitOffset += FixedTable[p] >> 16;
					int v = FixedTable[p] & unchecked((int)(0xFFFF));
					codeLengthCodeLengths[codeLenIdx] = v;
					if (v != 0)
					{
						space -= (32 >> v);
						numCodes++;
					}
				}
				ok = (numCodes == 1 || space == 0);
				ReadHuffmanCodeLengths(codeLengthCodeLengths, alphabetSize, codeLengths, br);
			}
			if (!ok)
			{
				throw new Org.Brotli.Dec.BrotliRuntimeException("Can't readHuffmanCode");
			}
			// COV_NF_LINE
			Org.Brotli.Dec.Huffman.BuildHuffmanTable(table, offset, HuffmanTableBits, codeLengths, alphabetSize);
		}

		private static int DecodeContextMap(int contextMapSize, byte[] contextMap, Org.Brotli.Dec.BitReader br)
		{
			Org.Brotli.Dec.BitReader.ReadMoreInput(br);
			int numTrees = DecodeVarLenUnsignedByte(br) + 1;
			if (numTrees == 1)
			{
				Org.Brotli.Dec.Utils.FillWithZeroes(contextMap, 0, contextMapSize);
				return numTrees;
			}
			bool useRleForZeros = Org.Brotli.Dec.BitReader.ReadBits(br, 1) == 1;
			int maxRunLengthPrefix = 0;
			if (useRleForZeros)
			{
				maxRunLengthPrefix = Org.Brotli.Dec.BitReader.ReadBits(br, 4) + 1;
			}
			int[] table = new int[Org.Brotli.Dec.Huffman.HuffmanMaxTableSize];
			ReadHuffmanCode(numTrees + maxRunLengthPrefix, table, 0, br);
			for (int i = 0; i < contextMapSize; )
			{
				Org.Brotli.Dec.BitReader.ReadMoreInput(br);
				Org.Brotli.Dec.BitReader.FillBitWindow(br);
				int code = ReadSymbol(table, 0, br);
				if (code == 0)
				{
					contextMap[i] = 0;
					i++;
				}
				else if (code <= maxRunLengthPrefix)
				{
					int reps = (1 << code) + Org.Brotli.Dec.BitReader.ReadBits(br, code);
					while (reps != 0)
					{
						if (i >= contextMapSize)
						{
							throw new Org.Brotli.Dec.BrotliRuntimeException("Corrupted context map");
						}
						// COV_NF_LINE
						contextMap[i] = 0;
						i++;
						reps--;
					}
				}
				else
				{
					contextMap[i] = unchecked((byte)(code - maxRunLengthPrefix));
					i++;
				}
			}
			if (Org.Brotli.Dec.BitReader.ReadBits(br, 1) == 1)
			{
				InverseMoveToFrontTransform(contextMap, contextMapSize);
			}
			return numTrees;
		}

		private static void DecodeBlockTypeAndLength(Org.Brotli.Dec.State state, int treeType)
		{
			Org.Brotli.Dec.BitReader br = state.br;
			int[] ringBuffers = state.blockTypeRb;
			int offset = treeType * 2;
			Org.Brotli.Dec.BitReader.FillBitWindow(br);
			int blockType = ReadSymbol(state.blockTypeTrees, treeType * Org.Brotli.Dec.Huffman.HuffmanMaxTableSize, br);
			state.blockLength[treeType] = ReadBlockLength(state.blockLenTrees, treeType * Org.Brotli.Dec.Huffman.HuffmanMaxTableSize, br);
			if (blockType == 1)
			{
				blockType = ringBuffers[offset + 1] + 1;
			}
			else if (blockType == 0)
			{
				blockType = ringBuffers[offset];
			}
			else
			{
				blockType -= 2;
			}
			if (blockType >= state.numBlockTypes[treeType])
			{
				blockType -= state.numBlockTypes[treeType];
			}
			ringBuffers[offset] = ringBuffers[offset + 1];
			ringBuffers[offset + 1] = blockType;
		}

		private static void DecodeLiteralBlockSwitch(Org.Brotli.Dec.State state)
		{
			DecodeBlockTypeAndLength(state, 0);
			int literalBlockType = state.blockTypeRb[1];
			state.contextMapSlice = literalBlockType << LiteralContextBits;
			state.literalTreeIndex = state.contextMap[state.contextMapSlice] & unchecked((int)(0xFF));
			state.literalTree = state.hGroup0.trees[state.literalTreeIndex];
			int contextMode = state.contextModes[literalBlockType];
			state.contextLookupOffset1 = Org.Brotli.Dec.Context.LookupOffsets[contextMode];
			state.contextLookupOffset2 = Org.Brotli.Dec.Context.LookupOffsets[contextMode + 1];
		}

		private static void DecodeCommandBlockSwitch(Org.Brotli.Dec.State state)
		{
			DecodeBlockTypeAndLength(state, 1);
			state.treeCommandOffset = state.hGroup1.trees[state.blockTypeRb[3]];
		}

		private static void DecodeDistanceBlockSwitch(Org.Brotli.Dec.State state)
		{
			DecodeBlockTypeAndLength(state, 2);
			state.distContextMapSlice = state.blockTypeRb[5] << DistanceContextBits;
		}

		private static void MaybeReallocateRingBuffer(Org.Brotli.Dec.State state)
		{
			int newSize = state.maxRingBufferSize;
			if ((long)newSize > state.expectedTotalSize)
			{
				/* TODO: Handle 2GB+ cases more gracefully. */
				int minimalNewSize = (int)state.expectedTotalSize + state.customDictionary.Length;
				while ((newSize >> 1) > minimalNewSize)
				{
					newSize >>= 1;
				}
				if (!state.inputEnd && newSize < 16384 && state.maxRingBufferSize >= 16384)
				{
					newSize = 16384;
				}
			}
			if (newSize <= state.ringBufferSize)
			{
				return;
			}
			int ringBufferSizeWithSlack = newSize + Org.Brotli.Dec.Dictionary.MaxTransformedWordLength;
			byte[] newBuffer = new byte[ringBufferSizeWithSlack];
			if (state.ringBuffer != null)
			{
				System.Array.Copy(state.ringBuffer, 0, newBuffer, 0, state.ringBufferSize);
			}
			else if (state.customDictionary.Length != 0)
			{
				/* Prepend custom dictionary, if any. */
				int length = state.customDictionary.Length;
				int offset = 0;
				if (length > state.maxBackwardDistance)
				{
					offset = length - state.maxBackwardDistance;
					length = state.maxBackwardDistance;
				}
				System.Array.Copy(state.customDictionary, offset, newBuffer, 0, length);
				state.pos = length;
				state.bytesToIgnore = length;
			}
			state.ringBuffer = newBuffer;
			state.ringBufferSize = newSize;
		}

		/// <summary>Reads next metablock header.</summary>
		/// <param name="state">decoding state</param>
		private static void ReadMetablockInfo(Org.Brotli.Dec.State state)
		{
			Org.Brotli.Dec.BitReader br = state.br;
			if (state.inputEnd)
			{
				state.nextRunningState = Org.Brotli.Dec.RunningState.Finished;
				state.bytesToWrite = state.pos;
				state.bytesWritten = 0;
				state.runningState = Org.Brotli.Dec.RunningState.Write;
				return;
			}
			// TODO: Reset? Do we need this?
			state.hGroup0.codes = null;
			state.hGroup0.trees = null;
			state.hGroup1.codes = null;
			state.hGroup1.trees = null;
			state.hGroup2.codes = null;
			state.hGroup2.trees = null;
			Org.Brotli.Dec.BitReader.ReadMoreInput(br);
			DecodeMetaBlockLength(br, state);
			if (state.metaBlockLength == 0 && !state.isMetadata)
			{
				return;
			}
			if (state.isUncompressed || state.isMetadata)
			{
				Org.Brotli.Dec.BitReader.JumpToByteBoundary(br);
				state.runningState = state.isMetadata ? Org.Brotli.Dec.RunningState.ReadMetadata : Org.Brotli.Dec.RunningState.CopyUncompressed;
			}
			else
			{
				state.runningState = Org.Brotli.Dec.RunningState.CompressedBlockStart;
			}
			if (state.isMetadata)
			{
				return;
			}
			state.expectedTotalSize += state.metaBlockLength;
			if (state.ringBufferSize < state.maxRingBufferSize)
			{
				MaybeReallocateRingBuffer(state);
			}
		}

		private static void ReadMetablockHuffmanCodesAndContextMaps(Org.Brotli.Dec.State state)
		{
			Org.Brotli.Dec.BitReader br = state.br;
			for (int i = 0; i < 3; i++)
			{
				state.numBlockTypes[i] = DecodeVarLenUnsignedByte(br) + 1;
				state.blockLength[i] = 1 << 28;
				if (state.numBlockTypes[i] > 1)
				{
					ReadHuffmanCode(state.numBlockTypes[i] + 2, state.blockTypeTrees, i * Org.Brotli.Dec.Huffman.HuffmanMaxTableSize, br);
					ReadHuffmanCode(NumBlockLengthCodes, state.blockLenTrees, i * Org.Brotli.Dec.Huffman.HuffmanMaxTableSize, br);
					state.blockLength[i] = ReadBlockLength(state.blockLenTrees, i * Org.Brotli.Dec.Huffman.HuffmanMaxTableSize, br);
				}
			}
			Org.Brotli.Dec.BitReader.ReadMoreInput(br);
			state.distancePostfixBits = Org.Brotli.Dec.BitReader.ReadBits(br, 2);
			state.numDirectDistanceCodes = NumDistanceShortCodes + (Org.Brotli.Dec.BitReader.ReadBits(br, 4) << state.distancePostfixBits);
			state.distancePostfixMask = (1 << state.distancePostfixBits) - 1;
			int numDistanceCodes = state.numDirectDistanceCodes + (48 << state.distancePostfixBits);
			// TODO: Reuse?
			state.contextModes = new byte[state.numBlockTypes[0]];
			for (int i = 0; i < state.numBlockTypes[0]; )
			{
				/* Ensure that less than 256 bits read between readMoreInput. */
				int limit = System.Math.Min(i + 96, state.numBlockTypes[0]);
				for (; i < limit; ++i)
				{
					state.contextModes[i] = unchecked((byte)(Org.Brotli.Dec.BitReader.ReadBits(br, 2) << 1));
				}
				Org.Brotli.Dec.BitReader.ReadMoreInput(br);
			}
			// TODO: Reuse?
			state.contextMap = new byte[state.numBlockTypes[0] << LiteralContextBits];
			int numLiteralTrees = DecodeContextMap(state.numBlockTypes[0] << LiteralContextBits, state.contextMap, br);
			state.trivialLiteralContext = true;
			for (int j = 0; j < state.numBlockTypes[0] << LiteralContextBits; j++)
			{
				if (state.contextMap[j] != j >> LiteralContextBits)
				{
					state.trivialLiteralContext = false;
					break;
				}
			}
			// TODO: Reuse?
			state.distContextMap = new byte[state.numBlockTypes[2] << DistanceContextBits];
			int numDistTrees = DecodeContextMap(state.numBlockTypes[2] << DistanceContextBits, state.distContextMap, br);
			Org.Brotli.Dec.HuffmanTreeGroup.Init(state.hGroup0, NumLiteralCodes, numLiteralTrees);
			Org.Brotli.Dec.HuffmanTreeGroup.Init(state.hGroup1, NumInsertAndCopyCodes, state.numBlockTypes[1]);
			Org.Brotli.Dec.HuffmanTreeGroup.Init(state.hGroup2, numDistanceCodes, numDistTrees);
			Org.Brotli.Dec.HuffmanTreeGroup.Decode(state.hGroup0, br);
			Org.Brotli.Dec.HuffmanTreeGroup.Decode(state.hGroup1, br);
			Org.Brotli.Dec.HuffmanTreeGroup.Decode(state.hGroup2, br);
			state.contextMapSlice = 0;
			state.distContextMapSlice = 0;
			state.contextLookupOffset1 = Org.Brotli.Dec.Context.LookupOffsets[state.contextModes[0]];
			state.contextLookupOffset2 = Org.Brotli.Dec.Context.LookupOffsets[state.contextModes[0] + 1];
			state.literalTreeIndex = 0;
			state.literalTree = state.hGroup0.trees[0];
			state.treeCommandOffset = state.hGroup1.trees[0];
			// TODO: == 0?
			state.blockTypeRb[0] = state.blockTypeRb[2] = state.blockTypeRb[4] = 1;
			state.blockTypeRb[1] = state.blockTypeRb[3] = state.blockTypeRb[5] = 0;
		}

		private static void CopyUncompressedData(Org.Brotli.Dec.State state)
		{
			Org.Brotli.Dec.BitReader br = state.br;
			byte[] ringBuffer = state.ringBuffer;
			// Could happen if block ends at ring buffer end.
			if (state.metaBlockLength <= 0)
			{
				Org.Brotli.Dec.BitReader.Reload(br);
				state.runningState = Org.Brotli.Dec.RunningState.BlockStart;
				return;
			}
			int chunkLength = System.Math.Min(state.ringBufferSize - state.pos, state.metaBlockLength);
			Org.Brotli.Dec.BitReader.CopyBytes(br, ringBuffer, state.pos, chunkLength);
			state.metaBlockLength -= chunkLength;
			state.pos += chunkLength;
			if (state.pos == state.ringBufferSize)
			{
				state.nextRunningState = Org.Brotli.Dec.RunningState.CopyUncompressed;
				state.bytesToWrite = state.ringBufferSize;
				state.bytesWritten = 0;
				state.runningState = Org.Brotli.Dec.RunningState.Write;
				return;
			}
			Org.Brotli.Dec.BitReader.Reload(br);
			state.runningState = Org.Brotli.Dec.RunningState.BlockStart;
		}

		private static bool WriteRingBuffer(Org.Brotli.Dec.State state)
		{
			/* Ignore custom dictionary bytes. */
			if (state.bytesToIgnore != 0)
			{
				state.bytesWritten += state.bytesToIgnore;
				state.bytesToIgnore = 0;
			}
			int toWrite = System.Math.Min(state.outputLength - state.outputUsed, state.bytesToWrite - state.bytesWritten);
			if (toWrite != 0)
			{
				System.Array.Copy(state.ringBuffer, state.bytesWritten, state.output, state.outputOffset + state.outputUsed, toWrite);
				state.outputUsed += toWrite;
				state.bytesWritten += toWrite;
			}
			return state.outputUsed < state.outputLength;
		}

		internal static void SetCustomDictionary(Org.Brotli.Dec.State state, byte[] data)
		{
			state.customDictionary = (data == null) ? new byte[0] : data;
		}

		/// <summary>Actual decompress implementation.</summary>
		internal static void Decompress(Org.Brotli.Dec.State state)
		{
			if (state.runningState == Org.Brotli.Dec.RunningState.Uninitialized)
			{
				throw new System.InvalidOperationException("Can't decompress until initialized");
			}
			if (state.runningState == Org.Brotli.Dec.RunningState.Closed)
			{
				throw new System.InvalidOperationException("Can't decompress after close");
			}
			Org.Brotli.Dec.BitReader br = state.br;
			int ringBufferMask = state.ringBufferSize - 1;
			byte[] ringBuffer = state.ringBuffer;
			while (state.runningState != Org.Brotli.Dec.RunningState.Finished)
			{
				switch (state.runningState)
				{
					case Org.Brotli.Dec.RunningState.BlockStart:
					{
						// TODO: extract cases to methods for the better readability.
						if (state.metaBlockLength < 0)
						{
							throw new Org.Brotli.Dec.BrotliRuntimeException("Invalid metablock length");
						}
						ReadMetablockInfo(state);
						/* Ring-buffer would be reallocated here. */
						ringBufferMask = state.ringBufferSize - 1;
						ringBuffer = state.ringBuffer;
						continue;
					}

					case Org.Brotli.Dec.RunningState.CompressedBlockStart:
					{
						ReadMetablockHuffmanCodesAndContextMaps(state);
						state.runningState = Org.Brotli.Dec.RunningState.MainLoop;
						goto case Org.Brotli.Dec.RunningState.MainLoop;
					}

					case Org.Brotli.Dec.RunningState.MainLoop:
					{
						// Fall through
						if (state.metaBlockLength <= 0)
						{
							state.runningState = Org.Brotli.Dec.RunningState.BlockStart;
							continue;
						}
						Org.Brotli.Dec.BitReader.ReadMoreInput(br);
						if (state.blockLength[1] == 0)
						{
							DecodeCommandBlockSwitch(state);
						}
						state.blockLength[1]--;
						Org.Brotli.Dec.BitReader.FillBitWindow(br);
						int cmdCode = ReadSymbol(state.hGroup1.codes, state.treeCommandOffset, br);
						int rangeIdx = (int)(((uint)cmdCode) >> 6);
						state.distanceCode = 0;
						if (rangeIdx >= 2)
						{
							rangeIdx -= 2;
							state.distanceCode = -1;
						}
						int insertCode = Org.Brotli.Dec.Prefix.InsertRangeLut[rangeIdx] + (((int)(((uint)cmdCode) >> 3)) & 7);
						int copyCode = Org.Brotli.Dec.Prefix.CopyRangeLut[rangeIdx] + (cmdCode & 7);
						state.insertLength = Org.Brotli.Dec.Prefix.InsertLengthOffset[insertCode] + Org.Brotli.Dec.BitReader.ReadBits(br, Org.Brotli.Dec.Prefix.InsertLengthNBits[insertCode]);
						state.copyLength = Org.Brotli.Dec.Prefix.CopyLengthOffset[copyCode] + Org.Brotli.Dec.BitReader.ReadBits(br, Org.Brotli.Dec.Prefix.CopyLengthNBits[copyCode]);
						state.j = 0;
						state.runningState = Org.Brotli.Dec.RunningState.InsertLoop;
						goto case Org.Brotli.Dec.RunningState.InsertLoop;
					}

					case Org.Brotli.Dec.RunningState.InsertLoop:
					{
						// Fall through
						if (state.trivialLiteralContext)
						{
							while (state.j < state.insertLength)
							{
								Org.Brotli.Dec.BitReader.ReadMoreInput(br);
								if (state.blockLength[0] == 0)
								{
									DecodeLiteralBlockSwitch(state);
								}
								state.blockLength[0]--;
								Org.Brotli.Dec.BitReader.FillBitWindow(br);
								ringBuffer[state.pos] = unchecked((byte)ReadSymbol(state.hGroup0.codes, state.literalTree, br));
								state.j++;
								if (state.pos++ == ringBufferMask)
								{
									state.nextRunningState = Org.Brotli.Dec.RunningState.InsertLoop;
									state.bytesToWrite = state.ringBufferSize;
									state.bytesWritten = 0;
									state.runningState = Org.Brotli.Dec.RunningState.Write;
									break;
								}
							}
						}
						else
						{
							int prevByte1 = ringBuffer[(state.pos - 1) & ringBufferMask] & unchecked((int)(0xFF));
							int prevByte2 = ringBuffer[(state.pos - 2) & ringBufferMask] & unchecked((int)(0xFF));
							while (state.j < state.insertLength)
							{
								Org.Brotli.Dec.BitReader.ReadMoreInput(br);
								if (state.blockLength[0] == 0)
								{
									DecodeLiteralBlockSwitch(state);
								}
								int literalTreeIndex = state.contextMap[state.contextMapSlice + (Org.Brotli.Dec.Context.Lookup[state.contextLookupOffset1 + prevByte1] | Org.Brotli.Dec.Context.Lookup[state.contextLookupOffset2 + prevByte2])] & unchecked((int)(0xFF));
								state.blockLength[0]--;
								prevByte2 = prevByte1;
								Org.Brotli.Dec.BitReader.FillBitWindow(br);
								prevByte1 = ReadSymbol(state.hGroup0.codes, state.hGroup0.trees[literalTreeIndex], br);
								ringBuffer[state.pos] = unchecked((byte)prevByte1);
								state.j++;
								if (state.pos++ == ringBufferMask)
								{
									state.nextRunningState = Org.Brotli.Dec.RunningState.InsertLoop;
									state.bytesToWrite = state.ringBufferSize;
									state.bytesWritten = 0;
									state.runningState = Org.Brotli.Dec.RunningState.Write;
									break;
								}
							}
						}
						if (state.runningState != Org.Brotli.Dec.RunningState.InsertLoop)
						{
							continue;
						}
						state.metaBlockLength -= state.insertLength;
						if (state.metaBlockLength <= 0)
						{
							state.runningState = Org.Brotli.Dec.RunningState.MainLoop;
							continue;
						}
						if (state.distanceCode < 0)
						{
							Org.Brotli.Dec.BitReader.ReadMoreInput(br);
							if (state.blockLength[2] == 0)
							{
								DecodeDistanceBlockSwitch(state);
							}
							state.blockLength[2]--;
							Org.Brotli.Dec.BitReader.FillBitWindow(br);
							state.distanceCode = ReadSymbol(state.hGroup2.codes, state.hGroup2.trees[state.distContextMap[state.distContextMapSlice + (state.copyLength > 4 ? 3 : state.copyLength - 2)] & unchecked((int)(0xFF))], br);
							if (state.distanceCode >= state.numDirectDistanceCodes)
							{
								state.distanceCode -= state.numDirectDistanceCodes;
								int postfix = state.distanceCode & state.distancePostfixMask;
								state.distanceCode = (int)(((uint)state.distanceCode) >> state.distancePostfixBits);
								int n = ((int)(((uint)state.distanceCode) >> 1)) + 1;
								int offset = ((2 + (state.distanceCode & 1)) << n) - 4;
								state.distanceCode = state.numDirectDistanceCodes + postfix + ((offset + Org.Brotli.Dec.BitReader.ReadBits(br, n)) << state.distancePostfixBits);
							}
						}
						// Convert the distance code to the actual distance by possibly looking up past distances
						// from the ringBuffer.
						state.distance = TranslateShortCodes(state.distanceCode, state.distRb, state.distRbIdx);
						if (state.distance < 0)
						{
							throw new Org.Brotli.Dec.BrotliRuntimeException("Negative distance");
						}
						// COV_NF_LINE
						if (state.maxDistance != state.maxBackwardDistance && state.pos < state.maxBackwardDistance)
						{
							state.maxDistance = state.pos;
						}
						else
						{
							state.maxDistance = state.maxBackwardDistance;
						}
						state.copyDst = state.pos;
						if (state.distance > state.maxDistance)
						{
							state.runningState = Org.Brotli.Dec.RunningState.Transform;
							continue;
						}
						if (state.distanceCode > 0)
						{
							state.distRb[state.distRbIdx & 3] = state.distance;
							state.distRbIdx++;
						}
						if (state.copyLength > state.metaBlockLength)
						{
							throw new Org.Brotli.Dec.BrotliRuntimeException("Invalid backward reference");
						}
						// COV_NF_LINE
						state.j = 0;
						state.runningState = Org.Brotli.Dec.RunningState.CopyLoop;
						goto case Org.Brotli.Dec.RunningState.CopyLoop;
					}

					case Org.Brotli.Dec.RunningState.CopyLoop:
					{
						// fall through
						int src = (state.pos - state.distance) & ringBufferMask;
						int dst = state.pos;
						int copyLength = state.copyLength - state.j;
						if ((src + copyLength < ringBufferMask) && (dst + copyLength < ringBufferMask))
						{
							for (int k = 0; k < copyLength; ++k)
							{
								ringBuffer[dst++] = ringBuffer[src++];
							}
							state.j += copyLength;
							state.metaBlockLength -= copyLength;
							state.pos += copyLength;
						}
						else
						{
							for (; state.j < state.copyLength; )
							{
								ringBuffer[state.pos] = ringBuffer[(state.pos - state.distance) & ringBufferMask];
								state.metaBlockLength--;
								state.j++;
								if (state.pos++ == ringBufferMask)
								{
									state.nextRunningState = Org.Brotli.Dec.RunningState.CopyLoop;
									state.bytesToWrite = state.ringBufferSize;
									state.bytesWritten = 0;
									state.runningState = Org.Brotli.Dec.RunningState.Write;
									break;
								}
							}
						}
						if (state.runningState == Org.Brotli.Dec.RunningState.CopyLoop)
						{
							state.runningState = Org.Brotli.Dec.RunningState.MainLoop;
						}
						continue;
					}

					case Org.Brotli.Dec.RunningState.Transform:
					{
						if (state.copyLength >= Org.Brotli.Dec.Dictionary.MinWordLength && state.copyLength <= Org.Brotli.Dec.Dictionary.MaxWordLength)
						{
							int offset = Org.Brotli.Dec.Dictionary.OffsetsByLength[state.copyLength];
							int wordId = state.distance - state.maxDistance - 1;
							int shift = Org.Brotli.Dec.Dictionary.SizeBitsByLength[state.copyLength];
							int mask = (1 << shift) - 1;
							int wordIdx = wordId & mask;
							int transformIdx = (int)(((uint)wordId) >> shift);
							offset += wordIdx * state.copyLength;
							if (transformIdx < Org.Brotli.Dec.Transform.Transforms.Length)
							{
								int len = Org.Brotli.Dec.Transform.TransformDictionaryWord(ringBuffer, state.copyDst, Org.Brotli.Dec.Dictionary.GetData(), offset, state.copyLength, Org.Brotli.Dec.Transform.Transforms[transformIdx]);
								state.copyDst += len;
								state.pos += len;
								state.metaBlockLength -= len;
								if (state.copyDst >= state.ringBufferSize)
								{
									state.nextRunningState = Org.Brotli.Dec.RunningState.CopyWrapBuffer;
									state.bytesToWrite = state.ringBufferSize;
									state.bytesWritten = 0;
									state.runningState = Org.Brotli.Dec.RunningState.Write;
									continue;
								}
							}
							else
							{
								throw new Org.Brotli.Dec.BrotliRuntimeException("Invalid backward reference");
							}
						}
						else
						{
							// COV_NF_LINE
							throw new Org.Brotli.Dec.BrotliRuntimeException("Invalid backward reference");
						}
						// COV_NF_LINE
						state.runningState = Org.Brotli.Dec.RunningState.MainLoop;
						continue;
					}

					case Org.Brotli.Dec.RunningState.CopyWrapBuffer:
					{
						System.Array.Copy(ringBuffer, state.ringBufferSize, ringBuffer, 0, state.copyDst - state.ringBufferSize);
						state.runningState = Org.Brotli.Dec.RunningState.MainLoop;
						continue;
					}

					case Org.Brotli.Dec.RunningState.ReadMetadata:
					{
						while (state.metaBlockLength > 0)
						{
							Org.Brotli.Dec.BitReader.ReadMoreInput(br);
							// Optimize
							Org.Brotli.Dec.BitReader.ReadBits(br, 8);
							state.metaBlockLength--;
						}
						state.runningState = Org.Brotli.Dec.RunningState.BlockStart;
						continue;
					}

					case Org.Brotli.Dec.RunningState.CopyUncompressed:
					{
						CopyUncompressedData(state);
						continue;
					}

					case Org.Brotli.Dec.RunningState.Write:
					{
						if (!WriteRingBuffer(state))
						{
							// Output buffer is full.
							return;
						}
						if (state.pos >= state.maxBackwardDistance)
						{
							state.maxDistance = state.maxBackwardDistance;
						}
						state.pos &= ringBufferMask;
						state.runningState = state.nextRunningState;
						continue;
					}

					default:
					{
						throw new Org.Brotli.Dec.BrotliRuntimeException("Unexpected state " + state.runningState);
					}
				}
			}
			if (state.runningState == Org.Brotli.Dec.RunningState.Finished)
			{
				if (state.metaBlockLength < 0)
				{
					throw new Org.Brotli.Dec.BrotliRuntimeException("Invalid metablock length");
				}
				Org.Brotli.Dec.BitReader.JumpToByteBoundary(br);
				Org.Brotli.Dec.BitReader.CheckHealth(state.br, true);
			}
		}
	}
}
