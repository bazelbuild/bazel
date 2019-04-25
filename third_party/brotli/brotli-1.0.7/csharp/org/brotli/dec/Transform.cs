/* Copyright 2015 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/
namespace Org.Brotli.Dec
{
	/// <summary>Transformations on dictionary words.</summary>
	internal sealed class Transform
	{
		private readonly byte[] prefix;

		private readonly int type;

		private readonly byte[] suffix;

		internal Transform(string prefix, int type, string suffix)
		{
			this.prefix = ReadUniBytes(prefix);
			this.type = type;
			this.suffix = ReadUniBytes(suffix);
		}

		internal static byte[] ReadUniBytes(string uniBytes)
		{
			byte[] result = new byte[uniBytes.Length];
			for (int i = 0; i < result.Length; ++i)
			{
				result[i] = unchecked((byte)uniBytes[i]);
			}
			return result;
		}

		internal static readonly Org.Brotli.Dec.Transform[] Transforms = new Org.Brotli.Dec.Transform[] { new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, 
			Org.Brotli.Dec.WordTransformType.Identity, " "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst1, string.Empty), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " the "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity
			, string.Empty), new Org.Brotli.Dec.Transform("s ", Org.Brotli.Dec.WordTransformType.Identity, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " of "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.UppercaseFirst, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " and "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst2, string.Empty), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast1, string.Empty), new Org.Brotli.Dec.Transform(", ", Org.Brotli.Dec.WordTransformType.Identity, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity
			, ", "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseFirst, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " in "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.Identity, " to "), new Org.Brotli.Dec.Transform("e ", Org.Brotli.Dec.WordTransformType.Identity, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "\""), new Org.Brotli.Dec.Transform(string.Empty, 
			Org.Brotli.Dec.WordTransformType.Identity, "."), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "\">"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "\n"), new 
			Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast3, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "]"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.Identity, " for "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst3, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast2, string.Empty), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " a "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " that "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseFirst
			, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, ". "), new Org.Brotli.Dec.Transform(".", Org.Brotli.Dec.WordTransformType.Identity, string.Empty), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType
			.Identity, ", "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst4, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " with "), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "'"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " from "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity
			, " by "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst5, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst6, string.Empty), new Org.Brotli.Dec.Transform
			(" the ", Org.Brotli.Dec.WordTransformType.Identity, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast4, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.Identity, ". The "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " on "), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " as "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " is "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast7
			, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast1, "ing "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "\n\t"), new Org.Brotli.Dec.Transform(string.Empty
			, Org.Brotli.Dec.WordTransformType.Identity, ":"), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, ". "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "ed "), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst9, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitFirst7, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.OmitLast6, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "("), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, ", "), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast8, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " at "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.Identity, "ly "), new Org.Brotli.Dec.Transform(" the ", Org.Brotli.Dec.WordTransformType.Identity, " of "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast5, string.Empty), new Org.Brotli.Dec.Transform(
			string.Empty, Org.Brotli.Dec.WordTransformType.OmitLast9, string.Empty), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseFirst, ", "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst
			, "\""), new Org.Brotli.Dec.Transform(".", Org.Brotli.Dec.WordTransformType.Identity, "("), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.UppercaseFirst, "\">"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "=\""), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, "."), new Org.Brotli.Dec.Transform(".com/", 
			Org.Brotli.Dec.WordTransformType.Identity, string.Empty), new Org.Brotli.Dec.Transform(" the ", Org.Brotli.Dec.WordTransformType.Identity, " of the "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst
			, "'"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, ". This "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, ","), new Org.Brotli.Dec.Transform(".", Org.Brotli.Dec.WordTransformType
			.Identity, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, "("), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, "."), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, " not "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, "=\""), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "er "
			), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseAll, " "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "al "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType
			.UppercaseAll, string.Empty), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "='"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, "\""), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, ". "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, "("), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, 
			"ful "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseFirst, ". "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "ive "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.Identity, "less "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, "'"), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "est "), new Org.Brotli.Dec.Transform
			(" ", Org.Brotli.Dec.WordTransformType.UppercaseFirst, "."), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, "\">"), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, "='"
			), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, ","), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity, "ize "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType
			.UppercaseAll, "."), new Org.Brotli.Dec.Transform("\u00c2\u00a0", Org.Brotli.Dec.WordTransformType.Identity, string.Empty), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.Identity, ","), new Org.Brotli.Dec.Transform(string.Empty
			, Org.Brotli.Dec.WordTransformType.UppercaseFirst, "=\""), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, "=\""), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.Identity
			, "ous "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, ", "), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseFirst, "='"), new Org.Brotli.Dec.Transform(" ", 
			Org.Brotli.Dec.WordTransformType.UppercaseFirst, ","), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseAll, "=\""), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseAll, ", "), new Org.Brotli.Dec.Transform
			(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, ","), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, "("), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.
			UppercaseAll, ". "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseAll, "."), new Org.Brotli.Dec.Transform(string.Empty, Org.Brotli.Dec.WordTransformType.UppercaseAll, "='"), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType
			.UppercaseAll, ". "), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseFirst, "=\""), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType.UppercaseAll, "='"), new Org.Brotli.Dec.Transform(" ", Org.Brotli.Dec.WordTransformType
			.UppercaseFirst, "='") };

		internal static int TransformDictionaryWord(byte[] dst, int dstOffset, byte[] word, int wordOffset, int len, Org.Brotli.Dec.Transform transform)
		{
			int offset = dstOffset;
			// Copy prefix.
			byte[] @string = transform.prefix;
			int tmp = @string.Length;
			int i = 0;
			// In most cases tmp < 10 -> no benefits from System.arrayCopy
			while (i < tmp)
			{
				dst[offset++] = @string[i++];
			}
			// Copy trimmed word.
			int op = transform.type;
			tmp = Org.Brotli.Dec.WordTransformType.GetOmitFirst(op);
			if (tmp > len)
			{
				tmp = len;
			}
			wordOffset += tmp;
			len -= tmp;
			len -= Org.Brotli.Dec.WordTransformType.GetOmitLast(op);
			i = len;
			while (i > 0)
			{
				dst[offset++] = word[wordOffset++];
				i--;
			}
			if (op == Org.Brotli.Dec.WordTransformType.UppercaseAll || op == Org.Brotli.Dec.WordTransformType.UppercaseFirst)
			{
				int uppercaseOffset = offset - len;
				if (op == Org.Brotli.Dec.WordTransformType.UppercaseFirst)
				{
					len = 1;
				}
				while (len > 0)
				{
					tmp = dst[uppercaseOffset] & unchecked((int)(0xFF));
					if (tmp < unchecked((int)(0xc0)))
					{
						if (tmp >= 'a' && tmp <= 'z')
						{
							dst[uppercaseOffset] ^= unchecked((byte)32);
						}
						uppercaseOffset += 1;
						len -= 1;
					}
					else if (tmp < unchecked((int)(0xe0)))
					{
						dst[uppercaseOffset + 1] ^= unchecked((byte)32);
						uppercaseOffset += 2;
						len -= 2;
					}
					else
					{
						dst[uppercaseOffset + 2] ^= unchecked((byte)5);
						uppercaseOffset += 3;
						len -= 3;
					}
				}
			}
			// Copy suffix.
			@string = transform.suffix;
			tmp = @string.Length;
			i = 0;
			while (i < tmp)
			{
				dst[offset++] = @string[i++];
			}
			return offset - dstOffset;
		}
	}
}
