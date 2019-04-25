/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>
	/// Tests for
	/// <see cref="Transform"/>
	/// .
	/// </summary>
	public class TransformTest
	{
		private static long Crc64(byte[] data)
		{
			long crc = -1;
			for (int i = 0; i < data.Length; ++i)
			{
				long c = (crc ^ (long)(data[i] & unchecked((int)(0xFF)))) & unchecked((int)(0xFF));
				for (int k = 0; k < 8; k++)
				{
					c = ((long)(((ulong)c) >> 1)) ^ (-(c & 1L) & -3932672073523589310L);
				}
				crc = c ^ ((long)(((ulong)crc) >> 8));
			}
			return ~crc;
		}

		[NUnit.Framework.Test]
		public virtual void TestTrimAll()
		{
			byte[] output = new byte[2];
			byte[] input = new byte[] { 119, 111, 114, 100 };
			// "word"
			Org.Brotli.Dec.Transform transform = new Org.Brotli.Dec.Transform("[", Org.Brotli.Dec.WordTransformType.OmitFirst5, "]");
			Org.Brotli.Dec.Transform.TransformDictionaryWord(output, 0, input, 0, input.Length, transform);
			byte[] expectedOutput = new byte[] { 91, 93 };
			// "[]"
			NUnit.Framework.Assert.AreEqual(expectedOutput, output);
		}

		[NUnit.Framework.Test]
		public virtual void TestCapitalize()
		{
			byte[] output = new byte[8];
			byte[] input = new byte[] { 113, unchecked((byte)(-61)), unchecked((byte)(-90)), unchecked((byte)(-32)), unchecked((byte)(-92)), unchecked((byte)(-86)) };
			// "qæप"
			Org.Brotli.Dec.Transform transform = new Org.Brotli.Dec.Transform("[", Org.Brotli.Dec.WordTransformType.UppercaseAll, "]");
			Org.Brotli.Dec.Transform.TransformDictionaryWord(output, 0, input, 0, input.Length, transform);
			byte[] expectedOutput = new byte[] { 91, 81, unchecked((byte)(-61)), unchecked((byte)(-122)), unchecked((byte)(-32)), unchecked((byte)(-92)), unchecked((byte)(-81)), 93 };
			// "[QÆय]"
			NUnit.Framework.Assert.AreEqual(expectedOutput, output);
		}

		[NUnit.Framework.Test]
		public virtual void TestAllTransforms()
		{
			/* This string allows to apply all transforms: head and tail cutting, capitalization and
			turning to upper case; all results will be mutually different. */
			// "o123456789abcdef"
			byte[] testWord = new byte[] { 111, 49, 50, 51, 52, 53, 54, 55, 56, 57, 97, 98, 99, 100, 101, 102 };
			byte[] output = new byte[2259];
			int offset = 0;
			for (int i = 0; i < Org.Brotli.Dec.Transform.Transforms.Length; ++i)
			{
				offset += Org.Brotli.Dec.Transform.TransformDictionaryWord(output, offset, testWord, 0, testWord.Length, Org.Brotli.Dec.Transform.Transforms[i]);
				output[offset++] = unchecked((byte)(-1));
			}
			NUnit.Framework.Assert.AreEqual(output.Length, offset);
			NUnit.Framework.Assert.AreEqual(8929191060211225186L, Crc64(output));
		}
	}
}
