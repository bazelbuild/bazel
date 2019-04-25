/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>Utilities for building Huffman decoding tables.</summary>
	internal sealed class Huffman
	{
		/// <summary>
		/// Maximum possible Huffman table size for an alphabet size of 704, max code length 15 and root
		/// table bits 8.
		/// </summary>
		internal const int HuffmanMaxTableSize = 1080;

		private const int MaxLength = 15;

		/// <summary>Returns reverse(reverse(key, len) + 1, len).</summary>
		/// <remarks>
		/// Returns reverse(reverse(key, len) + 1, len).
		/// <p> reverse(key, len) is the bit-wise reversal of the len least significant bits of key.
		/// </remarks>
		private static int GetNextKey(int key, int len)
		{
			int step = 1 << (len - 1);
			while ((key & step) != 0)
			{
				step >>= 1;
			}
			return (key & (step - 1)) + step;
		}

		/// <summary>
		/// Stores
		/// <paramref name="item"/>
		/// in
		/// <c>table[0], table[step], table[2 * step] .., table[end]</c>
		/// .
		/// <p> Assumes that end is an integer multiple of step.
		/// </summary>
		private static void ReplicateValue(int[] table, int offset, int step, int end, int item)
		{
			do
			{
				end -= step;
				table[offset + end] = item;
			}
			while (end > 0);
		}

		/// <param name="count">histogram of bit lengths for the remaining symbols,</param>
		/// <param name="len">code length of the next processed symbol.</param>
		/// <returns>table width of the next 2nd level table.</returns>
		private static int NextTableBitSize(int[] count, int len, int rootBits)
		{
			int left = 1 << (len - rootBits);
			while (len < MaxLength)
			{
				left -= count[len];
				if (left <= 0)
				{
					break;
				}
				len++;
				left <<= 1;
			}
			return len - rootBits;
		}

		/// <summary>Builds Huffman lookup table assuming code lengths are in symbol order.</summary>
		internal static void BuildHuffmanTable(int[] rootTable, int tableOffset, int rootBits, int[] codeLengths, int codeLengthsSize)
		{
			int key;
			// Reversed prefix code.
			int[] sorted = new int[codeLengthsSize];
			// Symbols sorted by code length.
			// TODO: fill with zeroes?
			int[] count = new int[MaxLength + 1];
			// Number of codes of each length.
			int[] offset = new int[MaxLength + 1];
			// Offsets in sorted table for each length.
			int symbol;
			// Build histogram of code lengths.
			for (symbol = 0; symbol < codeLengthsSize; symbol++)
			{
				count[codeLengths[symbol]]++;
			}
			// Generate offsets into sorted symbol table by code length.
			offset[1] = 0;
			for (int len = 1; len < MaxLength; len++)
			{
				offset[len + 1] = offset[len] + count[len];
			}
			// Sort symbols by length, by symbol order within each length.
			for (symbol = 0; symbol < codeLengthsSize; symbol++)
			{
				if (codeLengths[symbol] != 0)
				{
					sorted[offset[codeLengths[symbol]]++] = symbol;
				}
			}
			int tableBits = rootBits;
			int tableSize = 1 << tableBits;
			int totalSize = tableSize;
			// Special case code with only one value.
			if (offset[MaxLength] == 1)
			{
				for (key = 0; key < totalSize; key++)
				{
					rootTable[tableOffset + key] = sorted[0];
				}
				return;
			}
			// Fill in root table.
			key = 0;
			symbol = 0;
			for (int len = 1, step = 2; len <= rootBits; len++, step <<= 1)
			{
				for (; count[len] > 0; count[len]--)
				{
					ReplicateValue(rootTable, tableOffset + key, step, tableSize, len << 16 | sorted[symbol++]);
					key = GetNextKey(key, len);
				}
			}
			// Fill in 2nd level tables and add pointers to root table.
			int mask = totalSize - 1;
			int low = -1;
			int currentOffset = tableOffset;
			for (int len = rootBits + 1, step = 2; len <= MaxLength; len++, step <<= 1)
			{
				for (; count[len] > 0; count[len]--)
				{
					if ((key & mask) != low)
					{
						currentOffset += tableSize;
						tableBits = NextTableBitSize(count, len, rootBits);
						tableSize = 1 << tableBits;
						totalSize += tableSize;
						low = key & mask;
						rootTable[tableOffset + low] = (tableBits + rootBits) << 16 | (currentOffset - tableOffset - low);
					}
					ReplicateValue(rootTable, currentOffset + (key >> rootBits), step, tableSize, (len - rootBits) << 16 | sorted[symbol++]);
					key = GetNextKey(key, len);
				}
			}
		}
	}
}
