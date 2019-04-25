/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>Contains a collection of huffman trees with the same alphabet size.</summary>
	internal sealed class HuffmanTreeGroup
	{
		/// <summary>The maximal alphabet size in this group.</summary>
		private int alphabetSize;

		/// <summary>Storage for Huffman lookup tables.</summary>
		internal int[] codes;

		/// <summary>
		/// Offsets of distinct lookup tables in
		/// <see cref="codes"/>
		/// storage.
		/// </summary>
		internal int[] trees;

		/// <summary>Initializes the Huffman tree group.</summary>
		/// <param name="group">POJO to be initialised</param>
		/// <param name="alphabetSize">the maximal alphabet size in this group</param>
		/// <param name="n">number of Huffman codes</param>
		internal static void Init(Org.Brotli.Dec.HuffmanTreeGroup group, int alphabetSize, int n)
		{
			group.alphabetSize = alphabetSize;
			group.codes = new int[n * Org.Brotli.Dec.Huffman.HuffmanMaxTableSize];
			group.trees = new int[n];
		}

		/// <summary>Decodes Huffman trees from input stream and constructs lookup tables.</summary>
		/// <param name="group">target POJO</param>
		/// <param name="br">data source</param>
		internal static void Decode(Org.Brotli.Dec.HuffmanTreeGroup group, Org.Brotli.Dec.BitReader br)
		{
			int next = 0;
			int n = group.trees.Length;
			for (int i = 0; i < n; i++)
			{
				group.trees[i] = next;
				Org.Brotli.Dec.Decode.ReadHuffmanCode(group.alphabetSize, group.codes, next, br);
				next += Org.Brotli.Dec.Huffman.HuffmanMaxTableSize;
			}
		}
	}
}
